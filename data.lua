require 'image'
local t = require 'transforms'
local fix = t.Fix()
local opt = opt
local cache_path = paths.concat(opt.cache,'cache.t7')

if not paths.filep(cache_path) then
   print('Preparing train val and meanstd cache.')
   local function ls(dir) return sys.split(sys.ls(dir:gsub(" ","\\ ")),'\n') end

   local classes = ls(paths.concat(opt.data,'train'))
   print(classes)

   local result = {}
   result.train = {}
   result.val = {}
   local ntr,nva =0,0
   for k,v in ipairs(classes) do
      result.train[k] = {}
      local files = ls(paths.concat(opt.data,'train',v))
      for l,w in ipairs(files) do
         table.insert(result.train[k],'train/' .. v .. '/' .. w)
         ntr = ntr + 1
      end
      result.val[k] = {}
      files = ls(paths.concat(opt.data,'val',v))
      for l,w in ipairs(files) do
         table.insert(result.val[k],'val/' .. v .. '/' .. w)
         nva = nva + 1
      end
   end
   result.classes = classes
   result.trainSize = ntr
   result.valSize = nva


   local nSamples = 1000
   print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')

   local avg_size = result.trainSize/#result.classes
   local classlist= {}
   for l,class in ipairs(result['train']) do
      local list = tnt.ListDataset{
         list = class,
         load = function(im)
            return fix(image.load(paths.concat(opt.data,im)):float())
         end,
      }
      classlist[#classlist+1] = list:shuffle(avg_size,true)
   end

   local iter =  tnt.ConcatDataset{datasets = classlist}:shuffle(nSamples,true):iterator()
   local tm = torch.Timer()
   local meanEstimate = {0,0,0}
   for img in iter() do
      for j=1,3 do
         meanEstimate[j] = meanEstimate[j] + img[j]:mean()
      end
   end
   for j=1,3 do
      meanEstimate[j] = meanEstimate[j] / nSamples
   end
   mean = meanEstimate

   print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
   local stdEstimate = {0,0,0}
   for img in iter() do
      for j=1,3 do
         stdEstimate[j] = stdEstimate[j] + img[j]:std()
      end
   end
   for j=1,3 do
      stdEstimate[j] = stdEstimate[j] / nSamples
   end
   std = stdEstimate

   local cache = {}
   cache.mean = mean
   cache.std = std

   result.meanstd = cache
   print(cache)
   print('Time to estimate:', tm:time().real)
   torch.save(cache_path,result)
end

local dataset = torch.load(cache_path)
local avg_size = math.floor(dataset.trainSize/#dataset.classes)

local function getIterator(mode, nEpoch)
   return tnt.ParallelDatasetIterator{
      nthread = opt.nDonkeys,
      init = function(threadid)
         tnt = require 'torchnet'
         require 'image'
         t = require 'transforms'
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
            }:transform{
               input =
                  mode == 'train' and
                     tnt.transform.compose{
                        t.Fix(),
                        t.Scale(256),
                        t.RandomCrop(224),
                        t.ColorNormalize(dataset.meanstd),
                        t.HorizontalFlip(0.5),
                     }
                  or mode == 'val' and
                     tnt.transform.compose{
                        t.Fix(),
                        t.Scale(256),
                        t.CenterCrop(224),
                        t.ColorNormalize(dataset.meanstd),
                     }
            }
            classes[#classes+1] = mode == 'train' and list:shuffle(avg_size,true) or list:shuffle()
         end
         return tnt.ConcatDataset{datasets = classes}:shuffle():batch(opt.batchSize,'skip-last')
      end,
   }
end

return getIterator
