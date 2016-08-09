local function logging(model, optimState, meters, loggers,n)
   meters.conf.normalized = true
   image.save(paths.concat(opt.save,'confusion_' .. n ..'.jpg'),
               image.scale(meters.conf:value():float(),1000,1000,'simple'))
   meters.conf.normalized = false

   loggers.test:style{'-','-','-'}
   loggers.test:plot()

   loggers.train:style{'-','-'}
   loggers.train:plot()

   loggers.full_train:style{'-'}
   loggers.full_train:plot()

   torch.save(paths.concat(opt.save,'optim_' .. n ..'.t7'), optimstate)
   torch.save(paths.concat(opt.save,'meters_' .. n ..'.t7'),meters)
   local savemodel = model:clone('weight','bias')
   torch.save(paths.concat(opt.save,'model_' .. n ..'.t7'), savemodel)
   collectgarbage()
   collectgarbage()
end

return logging
