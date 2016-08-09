require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'scalegrad' -- nn.ScaleGrad layer
require 'cunn'
tnt = require 'torchnet'
local utils = paths.dofile'utils.lua'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)
print(opt)

local regimes = {
   -- start, end,    LR,   WD,
   {  1,     3,   1e-2,   5e-4, },
   {  4,     6,   5e-3,   5e-4  },
   {  7,     9,   1e-3,   0 },
   { 44,     52,   5e-4,   0 },
   { 53,    1e8,   1e-4,   0 },
}


cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.cache)
paths.mkdir(opt.save)

local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    dampening = 0.0,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
}

local model, criterion = paths.dofile('model.lua')
print(model)
local getIterator = paths.dofile('data.lua')

local timers = {
   batchTimer = torch.Timer(),
   dataTimer = torch.Timer(),
   epochTimer = torch.Timer(),
}

local meters = {
   conf = tnt.ConfusionMeter{k = opt.nClasses},
   val = tnt.AverageValueMeter(),
   train = tnt.AverageValueMeter(),
   train_clerr = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   clerr = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   ap = tnt.APMeter(),
}

function meters:reset()
   self.conf:reset()
   self.val:reset()
   self.train:reset()
   self.train_clerr:reset()
   self.clerr:reset()
   self.ap:reset()
end

local loggers = {
   test = optim.Logger(opt.save,'test.log'),
   train = optim.Logger(opt.save,'train.log'),
   full_train = optim.Logger(opt.save,'full_train.log'),
}

loggers.test:setNames{'Test Loss', 'Test acc.', 'Test mAP'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss'}

loggers.test.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false

local engine = tnt.OptimEngine()

engine.hooks.onStartEpoch = function(state)
   state.iterator = getIterator('train',state.epoch)
   local epoch = state.epoch + 1
   if opt.LR == 0.0 then -- if manually specified
      for _, row in ipairs(regimes) do
         if epoch >= row[1] and epoch <= row[2] then
              state.config.learningRate = row[3]
              state.config.weightDecay = row[4]
              break
         end
      end
   end
   timers.epochTimer:reset()
end

local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()

engine.hooks.onSample = function(state)
   inputs:resize(state.sample.input:size()):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input = inputs
   state.sample.target = targets:squeeze()
   timers.dataTimer:stop()
end

engine.hooks.onForwardCriterion = function(state)
   if state.training then
      print(('Epoch: [%d][%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
             state.epoch+1, state.t, timers.batchTimer:time().real, state.criterion.output,
             state.config.learningRate, timers.dataTimer:time().real))
      timers.batchTimer:reset()
      meters.train:add(state.criterion.output)
      meters.train_clerr:add(state.network.output,state.sample.target)
      loggers.full_train:add{state.criterion.output}
   else
      meters.conf:add(state.network.output,state.sample.target)
      meters.clerr:add(state.network.output,state.sample.target)
      meters.val:add(state.criterion.output)
      local tar = torch.ByteTensor(#state.network.output):fill(0)
      for k=1,state.sample.target:size(1) do
         tar[k][state.sample.target[k]]=1
      end
      meters.ap:add(state.network.output,tar)
   end
end

engine.hooks.onUpdate = function()
   timers.dataTimer:reset()
   timers.dataTimer:resume()
end

engine.hooks.onEndEpoch = function(state)
   print("Epoch Train Loss:" ,meters.train:value(),"Total Epoch time: ",timers.epochTimer:time().real)
   loggers.train:add{meters.train:value(),meters.train_clerr:value()[1]}
   loggers.train:add{meters.train:value(),meters.train_clerr:value()[1]}
   meters:reset()
   engine:test{
      network = model,
      iterator = getIterator('val',0),
      criterion = criterion,
   }
   loggers.test:add{meters.val:value(),meters.clerr:value()[1],meters.ap:value():mean()}
   print("Validation Loss" , meters.val:value())
   print("Accuracy: Top 1%", meters.clerr:value()[1])
   print("mean AP:",meters.ap:value():mean())
   local y = optim.ConfusionMatrix(opt.nClasses)
   y.mat = meters.conf:value()
   print(y)
   local log = paths.dofile'log.lua'
   log(state.network, state.config, meters, loggers,state.epoch)
   print("Testing Finished")
   timers.epochTimer:reset()
   state.t = 0
end

engine:train{
   network = model,
   iterator = tnt.TableDataset{data= {1}}:iterator(),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = optimState,
   maxepoch = opt.nEpochs,
}
