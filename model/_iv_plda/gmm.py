
'''
Part of the code is drawn from 
https://github.com/lixucuhk/adversarial-attack-on-GMM-i-vector-based-speaker-verification-systems
Paper:
Adversarial Attacks on GMM i-vector based Speaker Verification Systems

I made some improvements by removing the naive loop in "Zeroth_First_Stats" 
and adding support for processing multiple frames in parallel in "ComponentLogLikelihood". 
These improvements greatly speed up the computation.
'''

import torch
import torch.nn.functional as F

import math


class FullGMM(object):
	def __init__(self, mdlfile, random=False, device="cpu"):

		self.device = device

		if random == True:
			self.num_gaussians = 2048
			self.dim = 60
			self.gconsts = torch.ones(self.num_gaussians, device=self.device)
			self.weights = torch.ones(self.num_gaussians, device=self.device)
			self.means_invcovars = torch.ones(self.num_gaussians, self.dim, device=self.device)
			self.invcovars = torch.ones(self.num_gaussians, self.dim, self.dim, device=self.device)
		else:
			rdfile = open(mdlfile, 'r')
			line = rdfile.readline()
			while line != '':
				if '<GCONSTS>' in line:
					# print('processing <GCONSTS>')
					gconsts = line.split()[2:-1]
					self.num_gaussians = len(gconsts)
					for i in range(self.num_gaussians):
						gconsts[i] = float(gconsts[i])
					self.gconsts = torch.tensor(gconsts, device=self.device)
					line = rdfile.readline()
				elif '<WEIGHTS>' in line:
					# print('processing <WEIGHTS>')
					weights = line.split()[2:-1]
					# if len(weights) != self.num_gaussians:
					# 	print('Dimension does not match between weights and gconsts.')
					# 	exit(1)
					for i in range(self.num_gaussians):
						weights[i] = float(weights[i])
					self.weights = torch.tensor(weights, device=self.device)
					line = rdfile.readline()
				elif '<MEANS_INVCOVARS>' in line:
					# print('processing <MEANS_INVCOVARS>')
					line = rdfile.readline()
					means_invcovars = []
					for i in range(self.num_gaussians):
						data = line.split(' ')[2:-1]
						for j in range(len(data)):
							data[j] = float(data[j])
						means_invcovars.append(data)
						line = rdfile.readline()
					self.dim = len(data)
					self.means_invcovars = torch.tensor(means_invcovars, device=self.device)            # (self.num_gaussians, self.dim)
					# print(self.means_invcovars.size())
				elif '<INV_COVARS>' in line:
					# print('processing <INV_COVARS>')
					self.invcovars = torch.zeros(self.num_gaussians, self.dim, self.dim, device=self.device)
					for i in range(self.num_gaussians):
						line = rdfile.readline()
						for j in range(self.dim):
							data = line.split(' ')[:-1]
							for k in range(len(data)):
								self.invcovars[i][j][k] = float(data[k])
								self.invcovars[i][k][j] = float(data[k])
							line = rdfile.readline()
					# for i in range(self.num_gaussians):
					# 	self.invcovars[i] = self.SymmetricMatrix(self.invcovars[i])
				else:
					line = rdfile.readline()
			rdfile.close()
		self.Means() # (self.num_gaussians, self.dim)

	def Means(self):
		# print('processing <Means>')
		self.means = torch.zeros(self.num_gaussians, self.dim, device=self.device)
		self.means = torch.matmul(torch.inverse(self.invcovars), self.means_invcovars.unsqueeze(-1)).squeeze(-1)
		# print(self.means.size())


	def SymmetricMatrix(self, matrix):
		num_row, num_col = matrix.size()
		new_matrix = matrix
		for i in range(num_row):
			for j in range(i+1, num_col):
				new_matrix[i][j] = matrix[j][i]

		return new_matrix
	
# 	def ComponentLogLikelihood(self, data):
# 		# loglike = torch.zeros(self.num_gaussians)
# 		loglike = torch.matmul(self.means_invcovars, data)
# 		# print('!!!!!!!!!!!!!!!!!!')
# 		# print(loglike)
# 		loglike -= 0.5*torch.matmul(torch.matmul(self.invcovars, data), data)
# 		# print('!!!!!!!!!!!!!!!!!!')
# 		# print(loglike)
# 		loglike += self.gconsts
# 		# print('!!!!!!!!!!!!!!!!!!')
# 		# print(loglike)
# 		# print('!!!!!!!!!!!!!!!!!!')

# 		return loglike

# 	def Posterior(self, data):
# 		post = F.softmax(self.ComponentLogLikelihood(data), -1)

# 		return post

	def ComponentLogLikelihood(self, data, gmm_frame_bs=200): # data: (T, dim)
		loglike = torch.matmul(self.means_invcovars.unsqueeze(0), data.unsqueeze(-1)).squeeze(-1) # (T, n_g)
		bs = gmm_frame_bs
		for T_i in range(math.ceil(loglike.shape[0] / bs)):
			s = T_i*bs
			e = (T_i+1)*bs
			# print(data.shape[0], T_i, s, e, data[s:e].shape)
			loglike[s:e, :] -= 0.5*torch.matmul(torch.matmul(self.invcovars.unsqueeze(0), data[s:e, :].unsqueeze(1).unsqueeze(-1)).squeeze(-1), 
							    data[s:e, :].unsqueeze(-1)).squeeze(-1) # (T, n_g)
		loglike += self.gconsts

		return loglike

	def Posterior(self, data, gmm_frame_bs=200):
		post = F.softmax(self.ComponentLogLikelihood(data, gmm_frame_bs), -1)

		return post

	def Zeroth_FirstCenter_Stats(self, data_seq):
		num_frame = len(data_seq)
		zeroth_stats = torch.zeros(self.num_gaussians, device=self.device)
		firstcenter_stats = torch.zeros(self.num_gaussians, self.dim, device=self.device)
		for i in range(num_frame):
			post = self.Posterior(data_seq[i])
			zeroth_stats += post
			firstcenter_stats += torch.mm(post.unsqueeze(-1), data_seq[i].unsqueeze(0))

		firstcenter_stats -= torch.mm(torch.diag(zeroth_stats), self.means)

		return zeroth_stats, firstcenter_stats

	# naive loop over the frames in the original code, too slow
# 	def Zeroth_First_Stats(self, data_seq):
# 		num_frame = len(data_seq)
# 		zeroth_stats = torch.zeros(self.num_gaussians, device=self.device)
# 		first_stats = torch.zeros(self.num_gaussians, self.dim, device=self.device)
# 		for i in range(num_frame):
# 			post = self.Posterior(data_seq[i])
# 			zeroth_stats += post
# 			first_stats += torch.mm(post.unsqueeze(-1), data_seq[i].unsqueeze(0))

# 		# firstcenter_stats -= torch.mm(torch.diag(zeroth_stats), self.means)

# 		return zeroth_stats, first_stats
	
	# improvment by SEC4SR (this repo)
	def Zeroth_First_Stats(self, data_seq, gmm_frame_bs=200):
		posts = self.Posterior(data_seq, gmm_frame_bs) # (T, n_g)
		zeroth_stats = torch.sum(posts, 0) # (n_g)
		first_stats = torch.sum(torch.matmul(posts.unsqueeze(-1), data_seq.unsqueeze(1)), 0)

		return zeroth_stats, first_stats

	def DRV_Ni(self, data, post):
		centered_data = -1*self.means+data # C*F
		sig_in_cent = torch.matmul(self.invcovars, centered_data.unsqueeze(-1)).squeeze(-1) # C*F
		const = torch.matmul(sig_in_cent.t(), post).unsqueeze(-1) # F*1
		Ni_drv = torch.matmul(const, post.unsqueeze(-1).t())-torch.matmul(sig_in_cent.t(), torch.diag(post))

		return Ni_drv

	def DRV_ui(self, Ni_drv, data, post):
		ui_drv = []
		for i in range(self.num_gaussians):
			drv = post[i]*torch.eye(self.dim, device=self.device)+torch.matmul(Ni_drv[:,i].unsqueeze(-1), data.unsqueeze(-1).t())
			ui_drv.append(drv)
		ui_drv = torch.cat(ui_drv, 1) # F*CF

		return ui_drv

	def ui(self, posts_seq, data_seq):
		ui = []
		for i in range(self.num_gaussians):
			uij = torch.matmul(data_seq.t(), posts_seq[i]) # F*1
			ui.append(uij.t())

		ui = torch.cat(ui, 0)

		return ui

	def post_seq(self, data_seq):
		posts = []
		for i in range(len(data_seq)):
			posts.append(self.Posterior(data_seq[i]))

		posts = torch.stack(posts, 0).t()

		return posts

	
	def to(self, device):

		if device == self.device:
			return

		self.device = device
		self.gconsts = self.gconsts.to(self.device)
		self.weights = self.weights.to(self.device)
		self.means_invcovars = self.means_invcovars.to(self.device)
		self.invcovars = self.invcovars.to(self.device)
		self.means = self.means.to(self.device)
