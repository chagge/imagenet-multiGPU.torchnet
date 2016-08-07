require 'nn'
local utils = paths.dofile'utils.lua'

local function createModel(nClasses)
   local model = nn.Sequential()

   model:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   model:add(nn.SpatialBatchNormalization(64,1e-3))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   model:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   model:add(nn.SpatialBatchNormalization(192,1e-3))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   model:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   model:add(nn.SpatialBatchNormalization(384,1e-3))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   model:add(nn.SpatialBatchNormalization(256,1e-3))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   model:add(nn.SpatialBatchNormalization(256,1e-3))
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   model:add(nn.View(256*6*6))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(256*6*6, 4096))
   model:add(nn.BatchNormalization(4096, 1e-3))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(4096, 4096))
   model:add(nn.BatchNormalization(4096, 1e-3))
   model:add(nn.ReLU())
   model:add(nn.Linear(4096,nClasses))
   model:add(nn.LogSoftMax())

   utils.FCinit(model)
   utils.testModel(model)
   utils.MSRinit(model)

   return model
end

return createModel
