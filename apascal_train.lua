require 'aPascal-data'
require 'nn'
require 'cutorch'
require 'cunn'
--require 'cudnn'
require 'loadcaffe'
require 'optim'
local matio = require 'matio'
fname = sys.fpath()
cmd = torch.CmdLine()
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
opt1 = cmd:parse(arg)


opt = lapp[[
      --batchSize       (default 32)      Sub-batch size
      --dataRoot        (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/)        Data root folder
      --imageRoot       (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/apascal_images/)       Image dir
      --testRoot      (default ./home/yang/abd/overfeat-torch/aPascal-aYahoo/ayahoo_test_images/)  test dir
]]


errLogger = optim.Logger(paths.concat(opt1.save, 'Train_error.log'))
valLogger = optim.Logger(paths.concat(opt1.save, 'Val_error.log'))
accLogger = optim.Logger(paths.concat(opt1.save, 'Test_accuracy.log'))
obj_err1 = optim.Logger(paths.concat(opt1.save, '/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-I/val_loss1.log'))

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
local txt = io.open('/home/yang/abd/overfeat-torch/aPascal-aYahoo/balance_weight.txt')
for line in txt:lines() do
    line = line:split(' ')
    balance_weights[#balance_weights+1] = {pl = tonumber(line[1]), nl = tonumber(line[2])}
end
txt:close()

bce_w = {0.0788,0.1890,0.0126,0.0808,0.0466,1.4088,0.0864,0.0344,0.9641,0.4505,0.1135,0.3156,0.2889,0.3561,0.3347,0.5063,
          0.7709,0.2662,0.3765,0.4598,0.2614,0.0533,0.0091,0.0133,0.1295,0.0378,0.1494,0.0769,0.0647,0.0404,0.0590,0.0221,
          0.0173,0.0397,0.0128,0.0089,0.0112,0.0378,0.0316,0.0490,0.0839,0.0597,0.0382,0.0030,0.0146,0.0094,0.0326,0.0105,0.0164,
          0.0316,0.0201,0.5149,0.2870,0.1076,0.0775,0.6880,0.1144,0.0801,0.0405,0.0188,0.0141,0.1935,0.0351,0.0022}

fc_19 = torch.load('/home/yang/abd/overfeat-torch/mAP_data/Curves/Same_weights/Adam/LR_1e-4/weights_1_19.t7')

local model, sgdState

    --------------------------------------------------------Building Model
    model = nn.Sequential()
    
    model = torch.load('/home/yang/abd/overfeat-torch/model.t7')
    model:remove(20) --= nil --Remove softmax
    model:remove(19) --= nil --Remove view
    model:remove(18) --= nil --Remove last  layer
    model:add(nn.View(-1):setNumInputDims(3)) 
    model:add(nn.Linear(4096,64))
    model:add(nn.Sigmoid())
    model:add(nn.SplitTable(2,64))
    model = model:float()
    model:cuda()
	--[[ the block freezes the convolutional layers. It doesn't fine-tune the conv layers ]]
  --[[  local count = 0
    for i , m in ipairs(model.modules) do
        if count == 7 then break end
        if torch.type(m):find('SpatialConvolutionMM') then
           m.accGradParameters = function() end
           m.updateParameters = function() end
           count = count + 1
        end
    end
   ]]
    model:cuda()
  
weights, gradients = model:getParameters()
    --------------------------------------------------------Parameter initialize
    sgdState = {
     
       learningRate   = 1e-4,
      
      
    }
  
--------------------------------------------------------Loss
loss1 = nn.ParallelCriterion()
for i = 1,64 do
    local pl = balance_weights[i].pl
    local nl = balance_weights[i].nl
    local bce = nn.BCECriterion()--torch.Tensor(opt.batchSize):fill( nl/pl ):float() ):cuda()
    loss1:add(bce)--,nl/nl+pl)
end
loss1:cuda()
--------------------------------------------------------Loss End


print(model)
function forwardBackward()
    model:training()
    gradients:zero()
 
    ims,labels = dataset:get_samples('train')
    local target = {} 
    ims = ims:cuda()
    for i = 1,64 do
       labels[i] = labels[i]:cuda() 
    end
 
    collectgarbage(); collectgarbage();
  
    
    local y = model:forward(ims)
   
    local loss_val = loss1:forward(y, labels)
     
    local df_dw = loss1:backward(y, labels)
   
    model:backward(ims, df_dw) 
    
    local loss_per_attribute = {}
    for i = 1,64 do
       loss_per_attribute[i] = loss1.criterions[i].output
    end

    return loss_val, loss_per_attribute, gradients, ims:size(1)
end

function eval( ims, labels )
   
    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    local pred = torch.CudaTensor(64,64):zero():float()
    collectgarbage(); collectgarbage();
    local y = model:forward( ims:cuda() )
    for label_i = 1,64 do
        local prediction = torch.gt(y[label_i]:float(), torch.Tensor(y[label_i]:size()):fill(0.5)):float()
        local correct = torch.eq( prediction, labels[label_i] ):float()
        local not_correct = torch.ne( prediction, labels[label_i] ):float()
        local tp = torch.eq( correct + labels[label_i], torch.Tensor(correct:size()):fill(2.0) ):sum() 
        local fp = torch.eq( not_correct + prediction, torch.Tensor(not_correct:size()):fill(2.0) ):sum() 
        true_positive[label_i] = true_positive[label_i] + tp
        true_negative[label_i] = true_negative[label_i] + correct:sum() - tp
        false_positive[label_i] = false_positive[label_i] + fp
        false_negative[label_i] = false_negative[label_i] + not_correct:sum() - fp
    end
    

    return ims:size(1), true_positive, true_negative, false_positive, false_negative,pred,loss
end
-------------------------------------------------------------------------------------------------------------
function eval_test( ims, labels )
    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    local pred = torch.CudaTensor(64):zero():float()
    collectgarbage(); collectgarbage();
    local y = model:forward( ims:cuda() )
    for label_i = 1,64 do
        local prediction = torch.gt(y[label_i]:float(), torch.Tensor(y[label_i]:size()):fill(0.5)):float()
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
        pred[{i}] = y[i][1]
    end
    return ims:size(1), true_positive, true_negative, false_positive, false_negative,pred:float()
end
function eval_sample()
    model:evaluate()

    ims,labels= dataset:get_samples('val')
    return eval(ims, labels)
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

    local flag,ims,labels
    local co = dataset.get_valid_sample_co()

    local total = 0
    
    local correct = torch.Tensor(64):zero()
    local prediction = torch.Tensor(6355,64):zero()
    local true_positive = torch.Tensor(64):zero()
    local true_negative = torch.Tensor(64):zero()
    local false_positive = torch.Tensor(64):zero()
    local false_negative = torch.Tensor(64):zero()
    local targets = torch.Tensor(64,6355):zero()
    while true do
        flag,ims,labels = coroutine.resume(co, dataset)
        if( ims == nil ) then
            break
        end
        local err = 0
      
        local ret = {eval(ims,labels)}
       
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
    local targets = torch.Tensor(64,val_size):zero()
    local index = 1
    while i <= val_size do
        ims, labels= dataset:get_val_samples(index)
       
        for j = 1,64 do
           targets[{{j},i}]:copy(labels[j])
        end
        local ret = {eval_test(ims,labels)}
       
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
-----------------------------------yahoo-----------------------------------------------------
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
        
        for j = 1,64 do
           targets[{{j},i}]:copy(labels[j])
        end
        local ret = {eval_test(ims,labels)}
        
        prediction[{{},i}]:copy(ret[6])
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
  matio.save('/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-I/Data_mAP_ft1_val'..tostring(i)..'.mat',{scores = prediction},labels = targets})
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
      
      optimizer(function() return loss_val, gradients end,
                       weights,
                       sgdState)
      err = err+loss_val
      sgdState.nSampledImages = sgdState.nSampledImages + batchProcessed
      sgdState.nEvalCounter = sgdState.nEvalCounter + 1
      xlua.progress(sgdState.nSampledImages%epochSize, epochSize)
      
      if math.floor(sgdState.nSampledImages / epochSize) ~= sgdState.epochCounter then
         -- Epoch completed!
         err = err/train_size
         --errLogger:add{['train loss']    = err}
         xlua.progress(sgdState.epochCounter, maxEpoch)
         sgdState.epochCounter = math.floor(sgdState.nSampledImages / epochSize)
         
	 print("\n\n----- Epoch "..sgdState.epochCounter.." -----")
         err = 0
        -- if sgdState.epochCounter == 3 or sgdState.epochCounter == 6 or sgdState.epochCounter == 10 then sgdState.learningRate = sgdState.learningRate*0.1 end
         if afterEpoch then afterEpoch(sgdState.epochCounter) end
         if sgdState.epochCounter > maxEpoch then
           
             break
         end
      end
   end
end


--------------------------------------------------------------------------------------
train( forwardBackward, weights, sgdState, train_size,14, afterEpoch )
---------------------------Testing---------------------------------------------------

print 'testing ayahoo'
total1,true_positive1,true_negative1,false_positive1,false_negative1,prediction1,targets1 = eval_all_yahoo()
matio.save('/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-I/Data_mAP_ft1_test.mat',{scores = prediction1,labels = targets1})