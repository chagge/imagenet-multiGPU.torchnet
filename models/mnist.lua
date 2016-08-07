require 'nn'
local utils = paths.dofile'utils.lua'

local function createModel(nClasses)
   local model = nn.Sequential()

   model:add(nn.View(32*32))
   -- model:add(nn.Linear(32*32, 32*32))
   model:add(nn.Linear(32*32, 10))
   -- model:add(nn.ReLU(true))
   -- utils.FCinit(model)
   -- utils.testModel(model)
   -- utils.MSRinit(model)

   return model
end

return createModel
