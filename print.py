import torch

for id in range(5):
	# 打印激活的scale zero
	quan_scale = torch.load("output_pt/input/input.{}.scale.pt".format(id))
	quan_zero = torch.load("output_pt/input/input.{}.zero.pt".format(id))
	print(id)
	print(quan_scale)
	print(quan_zero)

# for id in range(5):
# 	# 打印pe_out
# 	for pe_id in range(4):
# 		pe_out = torch.load("output_pt/pe_out/pe_output{}_{}.pt".format(id, pe_id))
# 		print(id, pe_id)
# 		print(pe_out)

# for id in range(5):
# 	# 打印pe_add
# 	pe_add = torch.load("output_pt/pe_add/pe_add_output{}.pt".format(id))
# 	print(id)
# 	print(pe_add)

# for id in range(5):
# 	# 打印pe_add
# 	requan_factor = torch.load("output_pt/requan_factor/requan_{}_{}.pt".format(id,id+1))
# 	requan_n = torch.load("output_pt/requan_factor/n_{}_{}.pt".format(id,id+1))
# 	print(id)
# 	print(requan_factor)
# 	print(requan_n)