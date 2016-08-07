--[[
Prepares temporary file with train val split and information.
Data is stored classwise, which is needed for in class balancing
meanstd file contains data about meanstd (for imagenet it is given)
]]

opt = lapp[[
   -s,--source    (default "source")   source that contains train val folders/images
   -n,--name      (default "name")     name of t7 file
   -m,--meanstd   (default "ilsvrc_2012.t7")     path of meanstdCache file
]]

local function ls(dir) return sys.split(sys.ls(dir:gsub(" ","\\ ")),'\n') end

classes = ls(paths.concat(opt.source,'train'))
print(classes)

local result = {}
result.train = {}
result.val = {}

local ntr,nva =0,0

for k,v in ipairs(classes) do
   result.train[k] = {}
   local files = ls(paths.concat(opt.source,'train',v))
   for l,w in ipairs(files) do
      table.insert(result.train[k],'train/' .. v .. '/' .. w)
      ntr = ntr + 1
   end
   result.val[k] = {}
   files = ls(paths.concat(opt.source,'val',v))
   for l,w in ipairs(files) do
      table.insert(result.val[k],'val/' .. v .. '/' .. w)
      nva = nva + 1
   end
end
result.classes = classes
result.meanstd = torch.load(opt.meanstd)
result.trainSize = ntr
result.valSize = nva
torch.save(opt.name..'.t7',result)
