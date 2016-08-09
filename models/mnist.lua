require 'nn'

local model = nn.Sequential()

model:add(nn.View(32*32*3))
model:add(nn.Linear(32*32*3, 10))

return model:cuda(), nn.CrossEntropyCriterion():cuda()
