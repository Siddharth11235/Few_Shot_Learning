import torch
import torch.nn as nn
import torch.nn.functional as F


class Siamese_Network(nn.Module):
	def __init__(self):
		super(Siamese_Network, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 64, 10),  # 64 at 96*96
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),  # 64 at 48*48
			nn.Conv2d(64, 128, 7),
			nn.ReLU(),    # 128 at 42*42
			nn.MaxPool2d(2),   # 128 at 21*21
			nn.Conv2d(128, 128, 4),
			nn.ReLU(), # 128 at 18*18
			nn.MaxPool2d(2), # 128 at 9*9
			nn.Conv2d(128, 256, 4),
			nn.ReLU(),   # 256 at 6*6
			)
		self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
		self.out = nn.Linear(4096, 1)

	def forward_conv(self, leg):
		leg =self.conv(leg)
		leg = leg.view(leg.size()[0], -1)
		leg = self.liner(leg)
		return leg

	def forward(self, left, right):
		outLeft = self.forward_conv(left)
		outRight = self.forward_conv(right)
		dis = torch.abs(outLeft - outRightt2)
		out = self.out(dis)
		#  return self.sigmoid(out)
		return out

# for test
if __name__ == '__main__':
    net = Siamese_Network()
    print(net)
    print(list(net.parameters()))