import torch

for id in range(5):
	# inp_max = torch.load("output_pt/input/input.{}.max_val.pt".format(id))
	# inp_min = torch.load("output_pt/input/input.{}.min_val.pt".format(id))
	# quan_max = 2 ** (QUAN_BIT - 1) - 1
	# quan_min = 0 - 2 ** (QUAN_BIT - 1)
	# quan_scale = (inp_max - inp_min) / (quan_max - quan_min)
	# quan_zero = quan_min - round(inp_min/quan_scale)
	quan_scale = torch.load("output_pt/input/input.{}.scale.pt".format(id))
	quan_zero = torch.load("output_pt/input/input.{}.zero.pt".format(id))
	print(id)
	print(quan_scale)
	print(quan_zero)