local ScaleGrad, parent = torch.class('nn.ScaleGrad', 'nn.Identity')

function ScaleGrad:__init(factor)
   parent.__init(self)
   self.factor = factor
end

function ScaleGrad:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput* self.factor
   return self.gradInput
end
