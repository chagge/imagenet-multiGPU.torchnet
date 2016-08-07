-- This file is specific to the model being loaded
require 'nn'
require 'utils/scalegrad'
local utils = paths.dofile'utils.lua'

do
   local ScaleGrad, parent = torch.class('nn.ScaleGrad', 'nn.Identity')

   function ScaleGrad:__init(factor)
      parent.__init(self)
      self.factor = factor
   end

   function ScaleGrad:updateGradInput(input, gradOutput)
      self.gradInput = gradOutput* self.factor
      return self.gradInput
   end
end

local function createModel(nClasses,factor)
   local model = torch.load('/home/karan/pretrained/alexnet_torch_cudnn.t7')
   model:get(2):remove(10)
   local final = nn.Sequential():add(model):add(nn.ScaleGrad(factor)):add(nn.Linear(4096,nClasses))
   utils.testModel(final)
   return final
end

return createModel
