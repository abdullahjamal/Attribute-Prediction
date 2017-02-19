require 'data_read'
require 'nn'
require 'cutorch'
require 'cunn'
require 'loadcaffe'
require 'optim'
tnt = require 'torchnet'
local matio = require 'matio'
fname = sys.fpath()
cmd = torch.CmdLine()
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
opt1 = cmd:parse(arg)


opt = lapp[[
      --batchSize       (default 20)      Sub-batch size
      --sd              (default 1)       standard deviation for rbf kernel
      --dataRoot        (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/)        Data root folder
      --imageRoot       (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/apascal_images/)       Image dir
      --testRoot      (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/ayahoo_test_images/)  test dir
]]

print(opt.batchSize)

errLogger = optim.Logger(paths.concat(opt1.save, 'Train_error_varII.log'))
valLogger = optim.Logger(paths.concat(opt1.save, 'Val_error_varII.log'))
accLogger = optim.Logger(paths.concat(opt1.save, 'Test_accuracy_varII.log'))
obj_err1 = optim.Logger(paths.concat(opt1.save, '/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-III/exp/val_loss.log'))


torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

dataset = Dataset('/home/yang/abd/overfeat-torch/aPascal-aYahoo/',
                  '/home/yang/abd/overfeat-torch/aPascal-aYahoo/apascal_images/',
                   '/home/yang/abd/overfeat-torch/aPascal-aYahoo/ayahoo_test_images/',
                   torch.Tensor({0.0,0,0,0,0}),1.0,
                   opt.batchSize
		   )
train_size, val_size,test_size = dataset:size()
balance_weights = {}

classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'}
local model, sgdState, att_model, cat_model,branches
do
 local HSICCriterion,parent = torch.class('nn.HSICCriterion','nn.Criterion')
 
function HSICCriterion:_init()
    parent:_init(self)
end

local rbf_kernel = function(input_rbf,input_rbf2)
      local gamma = 0.5/(opt.std*opt.std)
      local diff = torch.add(input_rbf,-input_rbf2)
      local y = (torch.reshape(diff,1,diff:size(1))) * (diff)
      local rbf = torch.exp(-y[1]/gamma)
      return rbf
end
local function kernel_phi(input_x) 
    
   local k = torch.zeros(input_x:size(1),input_x:size(1))
    local j = 1
    local i = 1
    for i = 1,input_x:size(1) do
       for j = 1,input_x:size(1) do
          if (i~=j) then
            k[{i,j}] = rbf_kernel(input_x[i],input_x[j])
          end
       end
    end
   return k   
end

local function kernel_theta(input_y)
    local k = torch.zeros(input_y:size(1),input_y:size(1))
    local j = 1
    local i = 1
    for i = 1,input_y:size(1) do
       for j = 1,input_y:size(1) do
          if (i~=j) then
            k[{i,j}] = rbf_kernel(input_y[i],input_y[j])
          end
       end
    end
   return k
end

local function grad_phi(input)
   local input1 = input[1]
   local input2 = input[2]
   local k_phi = kernel_phi(input1)
   local k_theta = kernel_theta(input2)
   local H = (torch.eye(opt.batchSize)+torch.Tensor(opt.batchSize,opt.batchSize):fill(-1/opt.batchSize))
   local temp = (H*k_theta*H) 
   local dl_dkphi = (1/(opt.batchSize-1)^2)*(temp:transpose(1,2))
   local dl_phi = torch.Tensor(opt.batchSize,4096):fill(0)
   local temp2 = torch.Tensor(1,4096):fill(0)
   for i = 1,opt.batchSize do
       local temp1 = torch.Tensor(1,4096):fill(0)
       for j = 1,opt.batchSize do
          dkphi_phi = (-1/opt.std)*((k_phi[{i,j}])*(input1[{i,{}}]-input1[{j,{}}])/(torch.dist(input1[{i,{}}],input1[{j,{}}])))
          temp2:copy(dkphi_phi)
          temp1= temp1 + (dl_dkphi[{i,j}]*temp2)
      end
      temp1 = 2*temp1           
      dl_phi[{i,{}}]=temp1
   end
   
   return dl_phi
 end

local function grad_theta(input)
   local input1 = input[1]
   local input2 = input[2]
   local k_phi = kernel_phi(input1)
   local k_theta = kernel_theta(input2)
   local H = (torch.eye(opt.batchSize)+torch.Tensor(opt.batchSize,opt.batchSize):fill(-1/opt.batchSize))
   local temp = (H*k_phi*H) 
   local dl_dktheta = (1/(opt.batchSize-1)^2)*(temp)
   local dl_theta = torch.Tensor(opt.batchSize,4096):fill(0)
   local temp2 = torch.Tensor(1,4096):fill(0)
   for i = 1,opt.batchSize do
       local temp1 = torch.Tensor(1,4096):fill(0)
       for j = 1,opt.batchSize do
          dktheta_theta = (1/opt.std)*((k_theta[{i,j}])*(input2[{i,{}}]-input2[{j,{}}])/(torch.dist(input2[{i,{}}],input2[{j,{}}])))
          temp2:copy(dktheta_theta)
          temp1= temp1 + (dl_dktheta[{i,j}]*temp2)
      end
      temp1 = 2*temp1           
      dl_theta[{i,{}}]=temp1
   end
   return dl_theta
end

function HSICCriterion:updateOutput(input,target)  
   local k_phi = kernel_phi(input[1])
   local k_theta = kernel_theta(input[2])
  local H = (torch.eye(opt.batchSize)+torch.Tensor(opt.batchSize,opt.batchSize):fill(-1/opt.batchSize))
  local a = k_phi*H
  local b = k_theta*H
  local hsic = (1/((opt.batchSize-1)*(opt.batchSize-1)))*(torch.trace(a*b))
  self.output = hsic
  return self.output
end

function HSICCriterion:updateGradInput(input,gradOutput)
  local grad1 = grad_phi(input)
  local grad2 = grad_theta(input)
  self.gradInput = {grad1,grad2}
  nn.utils.recursiveType(self.gradInput,'torch.CudaTensor')
  return self.gradInput
 end

end

    --------------------------------------------------------Building Model
    model = nn.Sequential()
    model = torch.load('/home/yang/abd/overfeat-torch/model.t7')
    model:remove(20) --= nil --Remove softmax
    model:remove(19) --= nil --Remove view
    model:remove(18) --= nil --Remove last linear layer
    model:add(nn.View(-1):setNumInputDims(3))
   
    branches = nn.ConcatTable()

    att_model = nn.Sequential()
    att_model:add(nn.Linear(4096,4096))
    att_model:add(nn.ReLU(true))
    att_model:add(nn.Linear(4096, 64))
    att_model:add(nn.Sigmoid())
    att_model:add(nn.SplitTable(2,64))
    att_model = att_model:float()

    
    cat_model = nn.Sequential()
    cat_model:add(nn.Linear(4096,4096))
    cat_model:add(nn.ReLU(true))
    cat_model:add(nn.Linear(4096, 20))
    cat_model = cat_model:float()
   
    branches:add(att_model)
    branches:add(cat_model)
    
    model:add(branches)
    model:cuda()

    --------------------------------------------------------Parameter initialize
    sgdState = {
       --- For Adam Optimization
       learningRate   = 1e-4,
      
    }
--------------------------------------------------------Loss
local ce = nn.CrossEntropyCriterion()
local hsic = nn.HSICCriterion()
loss2 = nn.ParallelCriterion()
loss1 = nn.ParallelCriterion()
for i = 1,64 do
    local bce = nn.BCECriterion()
    loss1:add(bce))
end
loss2:add(loss1,0.4):add(ce,0.6):add(hsic,0.001)
loss2:cuda()

--------------------------------------------------------Loss End

weights, gradients = model:getParameters()

print(model)

function forwardBackward()
    model:training()
    gradients:zero()
 
    ims,labels,classes = dataset:get_samples('train')
   
    local target = {}
    local input1 = {} 
    local tar = {}
    ims = ims:cuda()
    for i = 1,64 do
      labels[i] = labels[i]:cuda() 
    end
    classes = classes:cuda() 
    collectgarbage(); collectgarbage();
  
    local y1 = (model:get(19):get(1):get(1).output)
    local y2 = (model:get(19):get(2):get(1).output)
    

    
    target = {labels,classes}
    
    input1 = {y1,y2}
    local y = model:forward(ims)
    table.insert(y,input1)
    local loss_val = loss2:forward(y, target)
    
    local df_dw = loss2:backward(y, target)
 

    model:backward(ims, df_dw) 
    
    table.remove(y,4)
    table.remove(y,3)

    local loss_per_attribute = {}
    for i = 1,64 do
       loss_per_attribute[i] = loss1.criterions[i].output
    end
    
    return loss_val, loss_per_attribute, gradients, ims:size(1)
end

function eval( ims, labels,class )
    for i = 1,64 do
      labels[i] = labels[i]:cuda() 
    end
    class = class:cuda()
    local target = {labels,class}
    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    local pred = torch.Tensor(64):zero()
    collectgarbage(); collectgarbage();
    local y = model:forward( ims:cuda() )
    local loss = loss2:forward(y, target)
     for i = 1,64 do
      labels[i] = labels[i]:float() 
    end
    for label_i = 1,64 do
        local prediction = torch.gt(y[1][label_i]:float(), torch.Tensor(y[1][label_i]:size()):fill(0.5)):float()
        local correct = torch.eq( prediction, labels[label_i] ):float()
        local not_correct = torch.ne( prediction, labels[label_i] ):float()
        local tp = torch.eq( correct + labels[label_i], torch.Tensor(correct:size()):fill(2.0) ):sum() 
        local fp = torch.eq( not_correct + prediction, torch.Tensor(not_correct:size()):fill(2.0) ):sum() 
        true_positive[label_i] = true_positive[label_i] + tp
        true_negative[label_i] = true_negative[label_i] + correct:sum() - tp
        false_positive[label_i] = false_positive[label_i] + fp
        false_negative[label_i] = false_negative[label_i] + not_correct:sum() - fp
    end

    return ims:size(1), true_positive, true_negative, false_positive, false_negative,loss
end
-------------------------------------------------------------------------------------------------------------
function eval_test( ims, labels,size )
    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    local pred = torch.CudaTensor(64,size):zero():float()
    collectgarbage(); collectgarbage();
    local y = model:forward( ims:cuda() )
    --print(y)
    for label_i = 1,64 do
        local prediction = torch.gt(y[1][label_i]:float(), torch.Tensor(y[1][label_i]:size()):fill(0.5)):float()
        local correct = torch.eq( prediction, labels[label_i] ):float()
        local not_correct = torch.ne( prediction, labels[label_i] ):float()
        local tp = torch.eq( correct + labels[label_i], torch.Tensor(correct:size()):fill(2.0) ):sum() 
        local fp = torch.eq( not_correct + prediction, torch.Tensor(not_correct:size()):fill(2.0) ):sum() 
        true_positive[label_i] = true_positive[label_i] + tp
        true_negative[label_i] = true_negative[label_i] + correct:sum() - tp
        false_positive[label_i] = false_positive[label_i] + fp
        false_negative[label_i] = false_negative[label_i] + not_correct:sum() - fp
    end
    for i = 1,64 do
        pred[{i}] = y[1][i][1]--:float())
    end
    return ims:size(1), true_positive, true_negative, false_positive, false_negative,pred:float()
end
 
function eval_sample()
    model:evaluate()
     local i = 1 
      local err = 0 
    while i <= val_size do

    	ims,labels,class= dataset:get_samples('val')
        print 'x'
    	local target = {} 
        ims = ims:cuda()
         for i = 1,64 do
           labels[i] = labels[i]:cuda() 
          end
         class = class:cuda() 
        collectgarbage(); collectgarbage();
  
        local target = {labels,classes}
    
        local y = model:forward(ims)
   
        err = err + loss_val
        err = err/opt.batchSize
        obj_err1:add{['obj_validation']   = err }
        obj_err1:style{['obj_validation'] = '-'}
        obj_err1:plot()
        i = i + opt.batchSize
    end
        
    
end
--------------------------------------training set evaluation------------------------------
function eval_all_train() --Evaluate
    model:evaluate()

    local flag,ims,labels
    local co = dataset.get_train_sample_co()

    local total = 0

    local correct = torch.Tensor(64):zero()

    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()

    while true do
        flag,ims,labels = coroutine.resume(co, dataset)
        if( ims == nil ) then
            break
        end
        local ret = {eval(ims,labels)}

        total = total + ret[1]
        true_positive = true_positive + ret[2]
        true_negative = true_negative + ret[3]
        false_positive = false_positive + ret[4]
        false_negative = false_negative + ret[5]

        xlua.progress(total,train_size)
    end

    return total,true_positive,true_negative,false_positive,false_negative
end
---------------------------------validation-------------------------
function eval_all() --Evaluate
    model:evaluate()

    local flag,ims,labels,class
    local co = dataset.get_valid_sample_co()

    local total = 0

    local correct = torch.Tensor(64):zero()

    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()

    while true do
        flag,ims,labels,class = coroutine.resume(co, dataset)
        if( ims == nil ) then
            break
        end
         print(labels,class)
        local err = 0
        local ret = {eval(ims,labels,class)}
        err = err + ret[6]
        err = err/opt.batchSize
        obj_err1:add{['obj_validation']   = err }
        obj_err1:style{['obj_validation'] = '-'}
        obj_err1:plot()

        total = total + ret[1]
        true_positive = true_positive + ret[2]
        true_negative = true_negative + ret[3]
        false_positive = false_positive + ret[4]
        false_negative = false_negative + ret[5]

        xlua.progress(total,val_size)
    end

    return total,true_positive,true_negative,false_positive,false_negative
end

----------------------aPascal Test-----------------------------------------------
function eval_all_test() --Test
    model:evaluate()

    local flag,ims,labels
    
    local total = 0
    local i = 1
    local correct = torch.Tensor(64):zero()

    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    local prediction = torch.Tensor(64,val_size):zero()
    local targets = torch.Tensor(64,6355):zero()
    local index = 1
    while i <= val_size do
        ims, labels= dataset:get_val_samples(index)
        local ret = {eval_test(ims,labels,1)}
        prediction[{{},i}]:copy(ret[6])
        total = total + ret[1]
        true_positive = true_positive + ret[2]
        true_negative = true_negative + ret[3]
        false_positive = false_positive + ret[4]
        false_negative = false_negative + ret[5]
        index = index + 1
        i = i + 1
        xlua.progress(total,val_size)
    end

    return total,true_positive,true_negative,false_positive,false_negative,prediction,targets
end
---------------------------------------ayahoo--------------------------------------------
function eval_all_yahoo() --Test
    model:evaluate()

    local flag,ims,labels
    
    local total = 0
    local i = 1
    local correct = torch.Tensor(64):zero()

    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    local prediction = torch.Tensor(64,test_size):zero()
    local targets = torch.Tensor(64,2642):zero()
    local index = 1
    while i <= test_size do
        ims, labels= dataset:get_test_samples(index)
       
        local ret = {eval_test(ims,labels,1)}
        
        prediction[{{},i,}]:copy(ret[6])
        total = total + ret[1]
        true_positive = true_positive + ret[2]
        true_negative = true_negative + ret[3]
        false_positive = false_positive + ret[4]
        false_negative = false_negative + ret[5]
        index = index + 1
        i = i + 1
        xlua.progress(total,test_size)
    end

    return total,true_positive,true_negative,false_positive,false_negative,prediction,targets
end
-----------------------------------------Training error ----------------------------------
function eval_train_err()
    print 'evaluate training set....'
    local total,true_positive,true_negative,false_positive,false_negative = eval_all_train()
   local sum_acc = 0
   local mean = 0
   local err = 0
  for i = 1,64 do
   local accuracy = (true_positive[i] + true_negative[i]) / total
   local precision = true_positive[i] / (true_positive[i] + false_positive[i])
   local recall = true_positive[i] / (true_positive[i] + false_negative[i])
   local f1 = 2 * precision * recall / (precision + recall)
   sum_acc = sum_acc+accuracy
  end
    mean = sum_acc/64
    err = 1-mean
  return err
end
-----------------------------------------------------------------------
---------------------validation error calculation-------------------------------
function afterEpoch(i) 
    
 print 'evaluate validation set....'
  local total,true_positive,true_negative,false_positive,false_negative,prediction,targets = eval_all_test()
  matio.save('/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-III/exp/Data_mAP_ft1_val'..tostring(i)..'.mat',{scores = prediction})--labels = targets})
    print 'resume training....'
end
-----------------------------------Training------------------------------------------------
function train( fb, weights, sgdState, epochSize, maxEpoch, afterEpoch )
   sgdState.epochCounter = sgdState.epochCounter or 0
   sgdState.nSampledImages = sgdState.nSampledImages or 0
   sgdState.nEvalCounter = sgdState.nEvalCounter or 0

  
     optimizer= optim.adam
  
    local err = 0
   while true do -- Each epoch 
      collectgarbage(); collectgarbage()
      local loss_val, loss_per_attribute, gradients, batchProcessed = fb()
      
      err = err+loss_val
      optimizer(function() return loss_val, gradients end,
                       weights,
                       sgdState)
      sgdState.nSampledImages = sgdState.nSampledImages + batchProcessed
      sgdState.nEvalCounter = sgdState.nEvalCounter + 1
      xlua.progress(sgdState.nSampledImages%epochSize, epochSize)
      
      if math.floor(sgdState.nSampledImages / epochSize) ~= sgdState.epochCounter then
         -- Epoch completed!
         err = err/train_size
         errLogger:add{['train loss']    = err}
        
         xlua.progress(sgdState.epochCounter, maxEpoch)
         sgdState.epochCounter = math.floor(sgdState.nSampledImages / epochSize)
         
	     print("\n\n----- Epoch "..sgdState.epochCounter.." -----")
         err = 0
        -- if sgdState.epochCounter == 9 then sgdState.learningRate = sgdState.learningRate*0.1 end
         if afterEpoch then afterEpoch(sgdState.epochCounter) end
         if sgdState.epochCounter > maxEpoch then
             break
         end
      end
   end
end

--------------------------------------------------------------------------------------
train( forwardBackward, weights, sgdState, train_size,14, afterEpoch )
---------------------------Testing--------------------4------------------------------
print 'testing ayahoo'
total1,true_positive1,true_negative1,false_positive1,false_negative1,prediction1,targets1 = eval_all_yahoo()
matio.save('/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-III/exp/Data_mAP_ft1_test.mat',{scores = prediction1})
