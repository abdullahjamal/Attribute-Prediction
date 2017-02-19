require 'pl'
require 'image'
stringx.import()

local Dataset = torch.class('Dataset')

function load_txt(filename,image_dir,zero_as_minus_one)
    local data = {}
    local txt = io.open(filename)
    for line in txt:lines() do
        line = line:split(' ')

        local instance = {}
        instance.fileloc = image_dir..line[1]
        instance.class = line[2]
        
        instance.crop = {lu = {x = tonumber(line[3]), y = tonumber(line[4]) },
                         rl = {x = tonumber(line[5]), y = tonumber(line[6]) } }
        instance.attribute = {}
        for i=7,70 do
            instance.attribute[i-6] = tonumber(line[i])
            if( zero_as_minus_one and instance.attribute[i-6] == 0) then
                instance.attribute[i-6] = -1
            end
        end
        data[#data+1] = instance
    end
    txt:close()
    
    return data
end
function Dataset:__init(path,image_dir,test_dir,mean,std,batch_size,loss_function)
    --local zero_as_minus_one = loss_function == 'margin'
    self.train_data = load_txt(path..'apascal_train1.txt',image_dir, zero_as_minus_one )
    self.val_data = load_txt(path..'apascal_test.txt',image_dir, zero_as_minus_one )
    self.test_data = load_txt(path..'ayahoo_test.txt',test_dir, zero_as_minus_one )
    self.mean = mean
    self.std = std
    self.batch_size = batch_size
    self.loss_function = loss_function
end

function Dataset:get_image_attribute(data,index)
    if( #data < index ) then
        print 'index error'
    end
    local channels = {'r','g','b'}
    local mean = {0.411,0.3812,0.3574}
    std = {0.2688,0.2582,0.2612}
    instance = data[index]
    if( instance.preprocessed == nil ) then
    --if( instance.resized == nil ) then
        local im = image.load(instance.fileloc)
        local cropped_image = image.crop(im, instance.crop.lu.x, instance.crop.lu.y, instance.crop.rl.x, instance.crop.rl.y )
       
        local resized = image.scale(cropped_image,224,224) 
        for i, channel in ipairs(channels) do
            resized[{ i,{},{} }]:add(-mean[i])
            resized[{ i,{},{} }]:div(std[i])
        end
        local preprocessed = resized
        instance.preprocessed = torch.Tensor(3,224,224):zero()

        instance.preprocessed:copy(preprocessed) 
   end
 
    return instance.preprocessed, instance.attribute,instance.class
end
function Dataset:get_image_attribute_test(data,index)
    if( #data < index ) then
        print 'index error'
    end
    local channels = {'r','g','b'}
    local mean = {0.411,0.3812,0.3574}
    local mean1 = {103.939,116.779,123.68}
    std = {0.2688,0.2582,0.2612} 
    
    instance = data[index]
    if( instance.preprocessed == nil ) then
    --if( instance.resized == nil ) then
        local im = image.load(instance.fileloc)
        
        local cropped_image = image.crop(im, instance.crop.lu.x, instance.crop.lu.y, instance.crop.rl.x, instance.crop.rl.y )
        local resized = image.scale(cropped_image,224,224)
        for i, channel in ipairs(channels) do
            resized[{ i,{},{} }]:add(-mean[i])
            resized[{ i,{},{} }]:div(std[i])
        end
        local preprocessed = resized
        instance.preprocessed = torch.Tensor(3,224,224):zero()

        instance.preprocessed:copy(preprocessed)
   end
    --print (instance.attribute)
    return instance.preprocessed, instance.attribute,instance.class
end

function Dataset:size()
    return #self.train_data, #self.val_data,#self.test_data
end
function Dataset:get_samples(train_or_val)
    local inputs = torch.Tensor(self.batch_size, 3, 224, 224):zero()
    local classes ={}
    local class1 = torch.Tensor(self.batch_size):zero()
    local labels = {}
    for i = 1,64 do
        labels[i] = torch.Tensor(self.batch_size):zero()
    end

    local data
    if( train_or_val == 'train' ) then
        data = self.train_data
    else
        data = self.val_data
    end

    for i = 1,self.batch_size do
        local im, attr,class = self:get_image_attribute(data, math.floor(math.random() * #data) + 1 )
        class1[{i}]= class
        inputs:select(1,i):copy( im )
        for label_i = 1,64 do
            labels[label_i][i] = attr[label_i]
        end
    end
    return inputs, labels,class1
end

function Dataset:get_test_samples(index)
  local inputs = torch.Tensor(1, 3, 224, 224):zero()
    local labels = {}
     for i = 1,64 do
        labels[i] = torch.Tensor(1):zero()
    end

   local data = self.test_data
   
    for i = 1,1 do
        local im, attr = self:get_image_attribute_test(data,index)
        inputs:select(1,i):copy( im )
        for label_i = 1,64 do
            labels[label_i][i] = attr[label_i]
        end
    end
    return inputs, labels
end
function Dataset:get_val_samples(index)
  local inputs = torch.Tensor(1, 3, 224, 224):zero()
    local labels = {}
     for i = 1,64 do
        labels[i] = torch.Tensor(1):zero()
    end

   local data = self.val_data
   
    for i = 1,1 do
        local im, attr = self:get_image_attribute(data, index)
        inputs:select(1,i):copy( im )
        for label_i = 1,64 do
            labels[label_i][i] = attr[label_i]
        end
    end
    return inputs, labels
end

function Dataset:get_train_sample_co()
    local co = coroutine.create( function(this)
        for i = 1, #this.train_data, this.batch_size do
            local remain = math.min(this.batch_size, #this.train_data-i)

            local inputs = torch.Tensor(remain, 3, 224, 224):zero()
            local labels = {}
            for label_i = 1,64 do
                labels[label_i] = torch.Tensor(remain):zero()
            end

            for j = 1,remain do
                local im, attr = this:get_image_attribute(this.train_data, i+j-1)
                inputs:select(1,j):copy(im)
                for label_i = 1,64 do
                    labels[label_i][j] = attr[label_i]
                end
            end

            coroutine.yield( inputs, labels )
        end
    end
    )
    return co
end

function Dataset:get_valid_sample_co()
    local co = coroutine.create( function(this)
        for i = 1, #this.val_data, this.batch_size do
            local remain = math.min(this.batch_size, #this.val_data-i)
            local inputs = torch.Tensor(remain, 3, 224, 224):zero()
            local labels = {}
            for label_i = 1,64 do
                labels[label_i] = torch.Tensor(remain):zero()
            end

            for j = 1,remain do
                local im, attr,class = this:get_image_attribute(this.val_data, i+j-1)
                inputs:select(1,j):copy(im)
                for label_i = 1,64 do
                    labels[label_i][j] = attr[label_i]
                end
            end

            coroutine.yield( inputs, labels )
        end
    end
    )
    return co
end
function Dataset:get_test_sample_co()
    local co = coroutine.create( function(this)
        for i = 1, #this.test_data, this.batch_size do
            local remain = math.min(this.batch_size, #this.test_data-i)

            local inputs = torch.Tensor(remain, 3, 224, 224):zero()
            local labels = {}
            for label_i = 1,64 do
                labels[label_i] = torch.Tensor(remain):zero()
            end

            for j = 1,remain do
                local im, attr = this:get_image_attribute(this.test_data, i+j-1)
                inputs:select(1,j):copy(im)
                for label_i = 1,64 do
                    labels[label_i][j] = attr[label_i]
                end
            end

            coroutine.yield( inputs, labels )
        end
    end
    )
    return co
end
