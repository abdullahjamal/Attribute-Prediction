require 'data_read'
require 'nn'
require 'cutorch'
require 'cunn'
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


errLogger = optim.Logger(paths.concat(opt1.save, 'Train_error_varII.log'))
valLogger = optim.Logger(paths.concat(opt1.save, 'Val_error_varII.log'))
accLogger = optim.Logger(paths.concat(opt1.save, 'Test_accuracy_varII.log'))
obj_err1 = optim.Logger(paths.concat(opt1.save, '/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-II/val_loss.log'))


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



local model, sgdState, att_model, cat_model,model1,model2,branches

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
    
    cat_model = nn.Sequential()
    cat_model:add(nn.Linear(4096,4096))
    cat_model:add(nn.ReLU(true)) 
    cat_model:add(nn.Linear(4096, 20))
    
   
    branches:add(att_model)
    branches:add(cat_model)
    
   
    model:add(branches)
   
   model:cuda()

   att_model:get(3).weight:copy(fc_38)


    --------------------------------------------------------Parameter initialize
    sgdState = {
       --- For Adam optimization ---
       learningRate   = 1e-4,
      
    }
--------------------------------------------------------Loss
local ce = nn.CrossEntropyCriterion()
loss2 = nn.ParallelCriterion()
loss1 = nn.ParallelCriterion()
for i = 1,64 do
    local pl = balance_weights[i].pl
    local nl = balance_weights[i].nl
    local bce = nn.BCECriterion()--torch.Tensor(opt.batchSize):fill( nl/pl ):float() ):cuda()
    loss1:add(bce)--,nl/nl+pl)
end
loss2:add(loss1):add(ce)
loss2:cuda()
--------------------------------------------------------Loss End

weights, gradients = model:getParameters()

function forwardBackward()
    model:training()
    gradients:zero()
 
    ims,labels,classes = dataset:get_samples('train')
    local target = {} 
    ims = ims:cuda()
    for i = 1,64 do
       labels[i] = labels[i]:cuda() 
    end
    classes = classes:cuda() 
    collectgarbage(); collectgarbage();
  
    target = {labels,classes}
    
    local y = model:forward(ims)
   
    local loss_val = loss2:forward(y, target)
     
    local df_dw = loss2:backward(y, target)
   
    model:backward(ims, df_dw) 
    
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
--------------------------------------trainingset evaluation------------------------------
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
   matio.save('/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-II/Data_mAP_ft1_val'..tostring(i)..'.mat',{scores = prediction})--labels = targets})
 
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
train( forwardBackward, weights, sgdState, train_size,10, afterEpoch )
---------------------------Testing--------------------------------------------------
print 'testing ayahoo'
total1,true_positive1,true_negative1,false_positive1,false_negative1,prediction1,targets1 = eval_all_yahoo()
matio.save('/home/yang/abd/overfeat-torch/mAP_data/Curves/Vgg/Var-II/Data_mAP_ft1_test.mat',{scores = prediction1})--,labels = targets1})
