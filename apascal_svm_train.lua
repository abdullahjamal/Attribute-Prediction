require 'aPascal-data'
require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'optim'
local matio = require 'matio'
fname = sys.fpath()
cmd = torch.CmdLine()
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
opt1 = cmd:parse(arg)

opt = lapp[[
      --batchSize       (default 64)      Sub-batch size
      --dataRoot        (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/)        Data root folder
      --imageRoot       (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/apascal_images/)       Image dir
      --testRoot      (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/ayahoo_test_images/)  test dir
]]

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

accLogger = optim.Logger(paths.concat(opt1.save, 'Test_svm.log'))

dataset = Dataset('/home/yang/abd/overfeat-torch/aPascal-aYahoo/',
                  '/home/yang/abd/overfeat-torch/aPascal-aYahoo/apascal_images/',
                   '/home/yang/abd/overfeat-torch/aPascal-aYahoo/ayahoo_test_images/',
                   torch.Tensor({0.0,0,0,0,0}),1.0,
                   opt.batchSize
		   )
train_size, val_size = dataset:size()
local model, sgdState

    --------------------------------------------------------Building Model
    model = nn.Sequential()
    model = torch.load('/home/yang/abd/overfeat-torch/model.t7')
    model:cuda()
    print(model)
    --------------------------------------------------------Parameter initialize
    sgdState = {
       --- For SGD with momentum ---
       learningRate   = 1e-3,
       learningRateDecay = 1e-7,
       weightDecay    = 1e-4,
       momentum     = 0.9,
      
    }

mean = torch.Tensor(1,3,231,231):float()

channels = {'r','g','b'}
mean = {0.411,0.3812,0.3574}
std = {0.2688,0.2582,0.2612}	
---------------------------------------
-- Feature extraction : Training images
---------------------------------------
local co = dataset:get_train_sample_co()
train_num, valid_num = dataset:size()
features = torch.Tensor(train_num,4096):float()
train_labels = {}
tr_label = torch.Tensor(train_num,64):float()
for label_i = 1,64 do
    train_labels[label_i] = torch.Tensor(train_num):zero():float()
end
index = 1
while true do
   flag,ims,target_labels = coroutine.resume(co,dataset)
   if( ims == nil ) then
       break
    end

   for i, channel in ipairs(channels) do
       ims[{ {},i,{},{} }]:add(-mean[i])
       ims[{ {},i,{},{} }]:div(std[i])
  end
    
   processed = ims
    collectgarbage();collectgarbage();
    ret = model:forward(processed:cuda())

    features:sub(index,index + ims:size()[1]-1):copy(model['modules'][16].output:float()) --FC7 features
    for label_i = 1,64 do
        train_labels[label_i]:sub(index,index + ims:size()[1]-1):copy(target_labels[label_i])
    end
    index = index + ims:size()[1]

    xlua.progress(index,train_num)
end

for i = 1,64 do
 tr_label[{{},i}]:copy(train_labels[i])
end
matio.save('train-fc7-features_mn.mat',{train_features = features})
matio.save('train-attribute-label_mn.mat',tr_label)
torch.save( '/home/yang/abd/overfeat-torch/experiments_svm/train-fc7-features_mn.th', features )
torch.save('/home/yang/abd/overfeat-torch/experiments_svm/train-attribute-labels_mn.th', train_labels )

-----------------------------------------
-- Save to appropriate format for SVM : Training Images
------------------------------------------
train_num, valid_num = dataset:size()
features = torch.load( '/home/yang/abd/overfeat-torch/experiments_svm/train-fc7-features_mn.th')
labels = torch.load( '/home/yang/abd/overfeat-torch/experiments_svm/train-attribute-labels_mn.th')
--
files = {}
for i = 1,64 do
  file = io.open( '/home/yang/abd/overfeat-torch/experiments_svm/attr'..tostring(i)..'.txt','w+')
  files[i] = file
end
for i = 1, train_num do
    local t = {}
   for j = 1,4096 do
        table.insert(t,tostring(j)..':'..tostring(features[i][j])..' ')
    end
    s = table.concat(t,'')..'\n'
    for label_i = 1, 64 do
        if( labels[label_i][i] == 0 ) then
            files[label_i]:write('-1'..' '..s)
        else
            files[label_i]:write('+1'..' '..s)
        end
    end
    xlua.progress(i,train_num)
end
for i = 1,64 do
    files[i]:close()
end

---------------------------------------
-- Feature extraction : Validation set images
---------------------------------------
--local co = dataset:get_valid_sample_co()
--train_num, valid_num = dataset:size()
--features = torch.Tensor(valid_num,4096):float()
--labels = {}
--for label_i = 1,64 do
  --  labels[label_i] = torch.Tensor(valid_num):zero():float()
--end
--
--index = 1
--while true do
  --  flag,ims,target_labels = coroutine.resume(co,dataset)
  --  if( ims == nil ) then
    --    break
  --  end

  --  processed = ims*255 - torch.expand(mean,ims:size()[1],3,231,231)
--
 --   collectgarbage();collectgarbage();
 --   ret = model:forward(processed:cuda())

 --   features:sub(index,index + ims:size()[1]-1):copy(model['modules'][16].output:float()) --FC7 features
 --   for label_i = 1,64 do
   --     labels[label_i]:sub(index,index + ims:size()[1]-1):copy(target_labels[label_i])
  --  end
 --   index = index + ims:size()[1]
--
   -- xlua.progress(index,valid_num)
--end
--torch.save('/home/dhill/abd/overfeat-torch/experiments_svm/valid-fc7-features.th', features )
--torch.save('/home/dhill/abd/overfeat-torch/experiments_svm/valid-attribute-labels.th', labels )

-----------------------------------------
-- Save to appropriate format for SVM : Validation set Images
-----------------------------------------
--train_num, valid_num = dataset:size()
--features = torch.load('/home/dhill/abd/overfeat-torch/experiments_svm/valid-fc7-features.th')
--labels = torch.load('/home/dhill/abd/overfeat-torch/experiments_svm/valid-attribute-labels.th')

--files = {}
--for i = 1,64 do
  --  file = io.open('/home/dhill/abd/overfeat-torch/experiments_svm/valid-attr'..tostring(i)..'.txt','w+')
  --  files[i] = file
--end
--for i = 1, valid_num do
  --  local t = {}
  --  for j = 1,4096 do
   --     table.insert(t,tostring(j)..':'..tostring(features[i][j])..' ')
   -- end
  --  s = table.concat(t,'')..'\n'
  --  for label_i = 1, 64 do
    --    if( labels[label_i][i] == 0 ) then
      --      files[label_i]:write('-1'..' '..s)
      --  else
        --    files[label_i]:write('+1'..' '..s)
     --   end
  --  end
  --  xlua.progress(i,valid_num)
--end
--for i = 1,64 do
  --  files[i]:close()
--end
--------------------------------------
-- Feature extraction : Testing set images
---------------------------------------
local co = dataset:get_test_sample_co()
train_num,valid_num,test_num = dataset:size()
print(test_num)
features = torch.Tensor(test_num,4096):float()
test_labels = {}
test_labels1 = torch.Tensor(test_num,64):float()
for label_i = 1,64 do
    test_labels[label_i] = torch.Tensor(test_num):zero():float()
end

index = 1
while true do
    flag,ims,target_labels = coroutine.resume(co,dataset)
    if( ims == nil ) then
        break
    end
   for i, channel in ipairs(channels) do
       ims[{ {},i,{},{} }]:add(-mean[i])
       ims[{ {},i,{},{} }]:div(std[i])
   end
    processed = ims
    collectgarbage();collectgarbage();
    ret = model:forward(processed:cuda())
    
    features:sub(index,index + ims:size()[1]-1):copy(model['modules'][16].output:float()) --FC7 features
    for label_i = 1,64 do
        test_labels[label_i]:sub(index,index + ims:size()[1]-1):copy(target_labels[label_i])
    end
    index = index + ims:size()[1]

    xlua.progress(index,test_num)
end
for i = 1,64 do
 test_labels1[{{},i}]:copy(test_labels[i])
end
matio.save('test-fc7-features_mn.mat',{test_features = features})
matio.save('test-attribute-label_mn.mat',test_labels1)
torch.save('/home/yang/abd/overfeat-torch/experiments_svm/test-fc7-features_mn.th', features )
torch.save('/home/yang/abd/overfeat-torch/experiments_svm/test-attribute-labels_mn.th', test_labels )

-----------------------------------------
-- Save to appropriate format for SVM : Testing set Images
-----------------------------------------
train_num, valid_num,test_num = dataset:size()
features = torch.load('/home/yang/abd/overfeat-torch/experiments_svm/test-fc7-features_mn.th')
labels = torch.load('/home/yang/abd/overfeat-torch/experiments_svm/test-attribute-labels_mn.th')

files = {}
for i = 1,64 do
    file = io.open('/home/yang/abd/overfeat-torch/experiments_svm/test-attr'..tostring(i)..'.txt','w+')
    files[i] = file
end
for i = 1, test_num do
    local t = {}
    for j = 1,4096 do
        table.insert(t,tostring(j)..':'..tostring(features[i][j])..' ')
    end
    s = table.concat(t,'')..'\n'
   for label_i = 1, 64 do
        if( labels[label_i][i] == 0 ) then
            files[label_i]:write('-1'..' '..s)
       else
            files[label_i]:write('+1'..' '..s)
        end
    end
    xlua.progress(i,test_num)
end
for i = 1,64 do
    files[i]:close()
end

---------------------------------------------------
-- Run SVM
---------------------------------------------------

true_positive = torch.Tensor(64):zero()
true_negative = torch.Tensor(64):zero()
false_positive = torch.Tensor(64):zero()
false_negative = torch.Tensor(64):zero()
accuracy_valid = torch.Tensor(64):zero()
accuracy_test = torch.Tensor(64):zero()
prediction = torch.Tensor(64,2642):zero()
svm = require 'svm'
for label_i = 1,64 do
   collectgarbage(); collectgarbage();
   train_data = svm.ascread('/home/dhill/abd/overfeat-torch/experiments_svm/attr'..tostring(label_i)..'.txt')
  -- valid_data = svm.ascread('/home/dhill/abd/overfeat-torch/experiments_svm/valid-attr'..tostring(label_i)..'.txt')
   test_data = svm.ascread('/home/dhill/abd/overfeat-torch/experiments_svm/test-attr'..tostring(label_i)..'.txt')
   model = liblinear.train(train_data) 

   labels_t,accuracy_t,dec_t = liblinear.predict(test_data,model)--['-b 1'])
   print(model)

   accuracy_test[{label_i}] = accuracy_t[1] 
  
   for i = 1, labels_t:size()[1] do
        if( labels_t[i] > 0 ) then
            if( test_data[i][1] > 0 ) then
                true_positive[label_i] = true_positive[label_i] + 1
            else
                false_positive[label_i] = false_positive[label_i] + 1
            end
        else
            if( test_data[i][1] > 0 ) then
                false_negative[label_i] = false_negative[label_i] + 1
            else
                true_negative[label_i] = true_negative[label_i] + 1
            end
        end
    end
    print'checking...'

    prediction[{label_i,{}}]:copy(dec_t)
    xlua.progress(label_i,64)
end


acc_t = (accuracy_test:sum())/64

print(acc_t)]]
matio.save('Data_auc_svm_dec.mat',{scores_svm = prediction})
