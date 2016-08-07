local tnt = require 'torchnet'
require 'image'
require 'optim'
require 'cunn'
require 'cudnn'; cudnn.benchmark = true
local optnet = require 'optnet'
local utils = paths.dofile'models/utils.lua'

local opt = lapp[[
   -b,--batchSize             (default 128)           batch size
   -c,--cache                 (default "../data/style.t7") t7 file
   -d,--data                  (default "/media/mathieu/ssd data/karan/images_small")  path to append to cache
   -f,--factor                (default 99)             factor for fine tuning
   -m,--model                 (default 'alexnet_init') path to file for loading model
   -r,--learningRate          (default 0.01)          learning rate
   -s,--save                  (default "logs")        subdirectory to save logs
   --momentum                 (default 0.9)           momentum
   --weightDecay              (default 0.0005)        weightDecay
   --dampening                (default 0)
   --nesterov                 (default true)
   --nGPU                     (default 1)
   --epoch_step               (default 5)             no. of epochs after which lr reduces
   --manualSeed               (default 1)
   --nThread                  (default 4)
   --maxepoch                 (default 100)           total epochs to run
]]

print(opt)

local dataset = torch.load(opt.cache)
local avg_size = math.floor(dataset.trainSize/#dataset.classes)
local epochSize = math.floor(dataset.trainSize/opt.batchSize)

paths.mkdir(opt.save)

local optimState = {
   learningRate = opt.learningRate,
   learningRateDecay = 0,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   dampening = opt.dampening,
   nesterov = opt.nesterov,
}
-----------
local model = opt.factor == 99 and dofile(opt.model)(#dataset.classes,opt.factor)
               or dofile(opt.model)(#dataset.classes)
cudnn.convert(model, cudnn)
model = utils.makeDataParallelTable(model, opt.nGPU)
model:cuda()
print(model)
local criterion = nn.CrossEntropyCriterion()
criterion:cuda()
-----------

local function getIterator(mode, nEpoch)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nThread,
      init = function(threadid)
         require 'torchnet'
         require 'image'
         t = require 'utils/transforms'
         torch.manualSeed(threadid+nEpoch+opt.manualSeed)
         pca = {                          -- from fb.resnet.torch
            eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
            eigvec = torch.Tensor{
               { -0.5675,  0.7192,  0.4009 },
               { -0.5808, -0.0045, -0.8140 },
               { -0.5836, -0.6948,  0.4203 },
            },
         }
      end,
      closure = function()
         classes= {}
         for l,class in ipairs(dataset[mode]) do
            local list = tnt.ListDataset{
               list = class,
               load = function(im)
                  return {
                     input = image.load(paths.concat(opt.data,im)):float(),
                     target = torch.LongTensor{l},
                  }
               end,
            }:transform{ -- imagetransformations
               input =
                  mode == 'train' and
                     tnt.transform.compose{
                        t.Fix(),
                        t.RandomSizedCrop(224),
                         t.ColorJitter({
                            brightness = 0.4,
                            contrast = 0.4,
                            saturation = 0.4,
                        }),
                        t.Lighting(0.1, pca.eigval, pca.eigvec),
                        t.ColorNormalize(dataset.meanstd),
                        t.HorizontalFlip(0.5),
                     }
                  or mode == 'val' and
                     tnt.transform.compose{
                        t.Fix(),
                        t.Scale(256),
                        t.ColorNormalize(dataset.meanstd),
                        t.CenterCrop(224),
                     }
            }
            classes[#classes+1] = mode == 'train' and list:shuffle(avg_size,true) or list:shuffle()
         end
         return tnt.ConcatDataset{datasets = classes}:shuffle():batch(opt.batchSize,'skip-last')
      end,
   }
end

local batchTimer = torch.Timer()
local dataTimer = torch.Timer()
local epochTimer = torch.Timer()

local meters = {
   conf = tnt.ConfusionMeter{k = #dataset.classes},
   val = tnt.AverageValueMeter(),
   train = tnt.AverageValueMeter(),
   clerr = tnt.ClassErrorMeter{topk = {1},accuracy=true},
   ap = tnt.APMeter(),
}

local logs = {
   train_loss_full = {},
   train_loss = {},
   val_loss = {},
   map = {},
   clerr = {},
}

function meters:reset()
   self.conf:reset()
   self.val:reset()
   self.train:reset()
   self.clerr:reset()
   self.ap:reset()
end

local engine = tnt.OptimEngine()

function log(model, optimstate, meters, logs)
   local gnuplot = require 'gnuplot'
   -- local savemodel = model:clone('weight','bias')
   meters.conf.normalized = true
   image.save(paths.concat(opt.save,'confusion_' .. #logs.train_loss ..'.jpg'),
               image.scale(meters.conf:value():float(),1000,1000,'simple'))
   meters.conf.normalized = false
   gnuplot.epsfigure(paths.concat(opt.save,'train_' .. #logs.train_loss ..'.eps'))
   gnuplot.plot('train',torch.Tensor(logs.train_loss_full),'-')
   gnuplot.plotflush()
   gnuplot.epsfigure(paths.concat(opt.save,'valtrain_' .. #logs.train_loss ..'.eps'))
   gnuplot.plot({'val',torch.Tensor(logs.val_loss)},
                {'train',torch.Tensor(logs.train_loss)})
   gnuplot.plotflush()
   gnuplot.epsfigure(paths.concat(opt.save,'map_' .. #logs.train_loss ..'.eps'))
   gnuplot.plot('train',torch.Tensor(logs.map),'-')
   gnuplot.plotflush()
   gnuplot.epsfigure(paths.concat(opt.save,'accuracy_' .. #logs.train_loss ..'.eps'))
   gnuplot.plot('train',torch.Tensor(logs.clerr),'-')
   gnuplot.plotflush()
   torch.save(paths.concat(opt.save,'optim_' .. #logs.train_loss ..'.t7'), optimstate)
   torch.save(paths.concat(opt.save,'meters_' .. #logs.train_loss ..'.t7'),meters)
   torch.save(paths.concat(opt.save,'logs_' .. #logs.train_loss ..'.t7'),logs)
   torch.save(paths.concat(opt.save,'model_' .. #logs.train_loss ..'.t7'), model)
end

engine.hooks.onStartEpoch = function(state)
   if (state.epoch+1) % opt.epoch_step == 0 then
      state.config.learningRate = state.config.learningRate / 10
      print("Learning Rate changed to", state.config.learningRate)
   end
   state.iterator = getIterator('train',state.epoch)
   epochTimer:reset()
end

local inputs = torch.CudaTensor()
local targets = torch.CudaTensor()

engine.hooks.onSample = function(state)
   inputs:resize(state.sample.input:size()):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input = inputs
   state.sample.target = targets:squeeze()
   dataTimer:stop()
end

engine.hooks.onForwardCriterion = function(state)
   if state.training then
      print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f LR %.0e DataLoadingTime %.3f'):format(
             state.epoch+1, state.t%epochSize + 1, epochSize, batchTimer:time().real, state.criterion.output,
             state.config.learningRate, dataTimer:time().real))
      batchTimer:reset()
      meters.train:add(state.criterion.output)
      table.insert(logs.train_loss_full,state.criterion.output)
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

engine.hooks.onEndEpoch = function(state)
   print("Epoch Train Loss:" ,meters.train:value(),"Total Epoch time: ",epochTimer:time().real)
   logs.train_loss[#logs.train_loss+1] = meters.train:value()
   meters:reset()
   epochTimer:reset()
   engine:test{
      network = model,
      iterator = getIterator('val',0),
      criterion = criterion,
   }
   logs.val_loss[#logs.val_loss+1] = meters.val:value()
   logs.clerr[#logs.clerr+1] = meters.clerr:value()[1]
   logs.map[#logs.map+1] = meters.ap:value():mean()
   print("Validation Loss" , meters.val:value())
   print("Accuracy: Top 1%", meters.clerr:value()[1])
   print("mean AP:",meters.ap:value():mean())
   local y = optim.ConfusionMatrix(#dataset.classes)
   y.mat = meters.conf:value()
   print(y)
   log(state.network, state.config, meters, logs)
   print("Testing Finished")
end

engine.hooks.onUpdate = function()
   dataTimer:reset()
   dataTimer:resume()
end

engine:train{
   network = model,
   iterator = tnt.TableDataset{data= {1}}:iterator(),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = optimState,
   maxepoch = opt.maxepoch,
}
