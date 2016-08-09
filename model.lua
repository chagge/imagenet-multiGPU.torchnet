require 'nn'
require 'cunn'
local utils = paths.dofile'utils.lua'

local model

if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   model = torch.load(opt.retrain):cuda()
   if opt.reset then
      model:get(2):remove(10)
      model:add(nn.ScaleGrad(opt.lr_factor):cuda())
      model:add(nn.Linear(4096,opt.nClasses):cuda())
   end
   model = utils.makeDataParallelTable(model, opt.nGPU) -- defined in util.lua
   utils.testModel(model)
   model:cuda()
else
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
   model = paths.dofile('models/' .. opt.netType .. '.lua')
   if opt.backend == 'cudnn' then
      require 'cudnn'
      cudnn.convert(model, cudnn)
   elseif opt.backend ~= 'nn' then
      error'Unsupported backend'
   end
   model = utils.makeDataParallelTable(model, opt.nGPU) -- defined in util.lua
end


local criterion = nn.CrossEntropyCriterion():cuda()

return model, criterion
