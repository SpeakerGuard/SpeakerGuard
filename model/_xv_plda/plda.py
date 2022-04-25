
'''
Part of the code is drawn from 
https://github.com/lixucuhk/adversarial-attack-on-GMM-i-vector-based-speaker-verification-systems
Paper:
Adversarial Attacks on GMM i-vector based Speaker Verification Systems

I have made some modifications to speed up the compuation by replacing naive loop (ComputeScores_loop) with matrix computation (ComputeScores)
Also, 'ComputeScores_loop' does not support batch_scoring, while 'ComputeScores' does.
'''

import torch
import kaldi_io

 
class PLDA(object):
	def __init__(self, mdlfile, random=False, device="cpu"):

		self.device = device

		if random == True:
			self.dim = 600
			self.mean = torch.ones(self.dim, device=self.device)
			self.transform = torch.ones(self.dim, self.dim, device=self.device)
			self.psi = torch.ones(self.dim, device=self.device)
		else:
			rdfile = open(mdlfile, 'r')
			line = rdfile.readline()
			data = line.split()[2:-1]
			self.dim = len(data)
			for i in range(self.dim):
				data[i] = float(data[i])
			self.mean = torch.tensor(data, device=self.device)

			line = rdfile.readline()
			line = rdfile.readline()
			transform_matrix = []
			for i in range(self.dim):
				data = line.split(' ')[2:-1]
				for j in range(self.dim):
					data[j] = float(data[j])
				transform_matrix.append(data)
				line = rdfile.readline()
			self.transform = torch.tensor(transform_matrix, device=self.device)

			data = line.split()[1:-1]
			for i in range(self.dim):
				data[i] = float(data[i])
			self.psi = torch.tensor(data, device=self.device)

			rdfile.close()

	def ReadIvectors(self, ivectorfile):
		keys = []
		data = []
		i = 0
		for key, mat in kaldi_io.read_vec_flt_scp(ivectorfile):
			# print(key)
			# print(mat)
			# print(len(mat.tolist()))
			# exit(0)
			i += 1
			keys.append(key) 
			data.append(mat.tolist())
		print('totally %d ivectors' %(i))
		return keys, data

	
	def Transform(self, ivector, num_examples, simple_length_norm, normalize_length):

		return self.TransformIvector(ivector, num_examples, simple_length_norm, normalize_length)

	def TransformIvector(self, ivector, num_examples, simple_length_norm, normalize_length):
		# print(self.transform, self.mean, ivector),
		trans_ivector = torch.matmul(self.transform, ivector-self.mean)
		factor = 1.0
		if simple_length_norm == True:
			factor = torch.sqrt(self.dim)/torch.norm(trans_ivector, 2)
		elif normalize_length == True:
			factor = self.GetNormalizaionFactor(trans_ivector, num_examples)

		# print('original ivector is \n')
		# print(trans_ivector)
		trans_ivector = trans_ivector*factor
		# print('factor is %f' %(factor))
		# print('transformed ivector is \n')
		# print(trans_ivector)

		return trans_ivector


	def GetNormalizaionFactor(self, trans_ivector, num_examples):
		trans_ivector_sq = torch.pow(trans_ivector, 2)
		inv_covar = 1.0/(self.psi + 1.0/num_examples)
		factor = torch.sqrt(self.dim / torch.dot(inv_covar, trans_ivector_sq))

		return factor

	def ComputeScores_loop(self, trans_trainivector, num_examples, trans_testivector):
		# trans_trainivector = self.TransformIvector(trainivector, num_examples, simple_length_norm, normalize_length)
		# trans_testivector = self.TransformIvector(testivector, 1, simple_length_norm, normalize_length)

		#### work out loglike_given_class
		# mean = torch.zeros(self.dim) 
		# variance = torch.zeros(self.dim)
		mean = torch.zeros(self.dim, device=self.device)
		variance = torch.zeros(self.dim, device=self.device)

		# debug
		# print(self.dim),
		# print(mean.size())
		# print(variance.size())
		# print(self.psi.size())
		# print(trans_trainivector.size())  
		# print(trans_testivector.size())    

		for i in range(self.dim):
			mean[i] = num_examples*self.psi[i]/(num_examples*self.psi[i]+1.0)*trans_trainivector[i]
			variance[i] = 1.0+self.psi[i]/(num_examples*self.psi[i]+1.0)

		logdet = torch.sum(torch.log(variance))

		sqdiff = torch.pow(trans_testivector-mean, 2)
		variance = 1.0/variance

		loglike_given_class = -0.5*(logdet + torch.log(2*torch.tensor(3.1415926, device=self.device))*self.dim + torch.dot(sqdiff, variance))

		### work out loglike_without_class
		sqdiff = torch.pow(trans_testivector, 2)
		variance = self.psi + 1.0
		logdet = torch.sum(torch.log(variance))
		variance = 1.0/variance
		loglike_without_class = -0.5*(logdet + torch.log(2*torch.tensor(3.1415926, device=self.device))*self.dim + torch.dot(sqdiff, variance))

		loglike_ratio = loglike_given_class - loglike_without_class

		return loglike_ratio

	## no for loop and batch train_ivectors version
	def ComputeScores(self, trans_trainivector, num_examples, trans_testivector):
		# trans_trainivector = self.TransformIvector(trainivector, num_examples, simple_length_norm, normalize_length)
		# trans_testivector = self.TransformIvector(testivector, 1, simple_length_norm, normalize_length)

		#### work out loglike_given_class
		# mean = torch.zeros(self.dim)
		# variance = torch.zeros(self.dim)
		# mean = torch.zeros(self.dim, device=self.device)
		# variance = torch.zeros(self.dim, device=self.device)
		n_train_ivectors = trans_trainivector.shape[0]
		assert num_examples == 1
		mean = torch.zeros((n_train_ivectors, self.dim), device=self.device)
		variance = torch.zeros((n_train_ivectors, self.dim), device=self.device)

		# debug
		# print(self.dim),  
		# print(mean.size()) 
		# print(variance.size()) 
		# print(self.psi.size()) 
		# print(trans_trainivector.size())  
		# print(trans_testivector.size())    

		# for i in range(self.dim):
		# 	mean[i] = num_examples*self.psi[i]/(num_examples*self.psi[i]+1.0)*trans_trainivector[i]
		# 	variance[i] = 1.0+self.psi[i]/(num_examples*self.psi[i]+1.0)
		# for i in range(self.dim):
		# 	mean[i] = num_examples*self.psi[i]/(num_examples*self.psi[i]+1.0)*trans_trainivector[i]
		# 	variance[i] = 1.0+self.psi[i]/(num_examples*self.psi[i]+1.0)
		mean = num_examples * self.psi / (num_examples * self.psi + 1.0) * trans_trainivector # (n, dim)
		variance = (1.0 + self.psi / (num_examples * self.psi + 1.0)).expand(n_train_ivectors, self.dim) # (n, dim)
		# print(mean.shape, variance.shape)

		# logdet = torch.sum(torch.log(variance)) 
		logdet = torch.sum(torch.log(variance), dim=1) # (n, ) 

		sqdiff = torch.pow(trans_testivector-mean, 2) # (n, dim)
		variance = 1.0/variance # (n, dim)

		# loglike_given_class = -0.5*(logdet + torch.log(2*torch.tensor(3.1415926, device=self.device))*self.dim + torch.dot(sqdiff, variance))
		loglike_given_class = -0.5*(logdet + torch.log(2*torch.tensor(3.1415926, device=self.device))*self.dim + torch.sum(sqdiff * variance, axis=1)) # (n, )

		### work out loglike_without_class
		sqdiff = torch.pow(trans_testivector, 2) # (dim, )
		variance = self.psi + 1.0 # (dim, )
		logdet = torch.sum(torch.log(variance)) # scalar
		variance = 1.0/variance # (dim, )
		loglike_without_class = -0.5*(logdet + torch.log(2*torch.tensor(3.1415926, device=self.device))*self.dim + torch.dot(sqdiff, variance)) # scalar

		loglike_ratio = loglike_given_class - loglike_without_class # (n,)

		return loglike_ratio

	def DRV_TransformIvector(self, ivector, num_examples, simple_length_norm, normalize_length):
		############ Currently we only consider simple_length_norm = False situation.
		if normalize_length == True:
			trans_ivector = torch.matmul(self.transform, ivector-self.mean)
			factor = 1.0
			factor = self.GetNormalizaionFactor(trans_ivector, num_examples)

			norm_drv = torch.zeros(self.dim, self.dim, device=self.device)
			trans_ivector_sq = torch.pow(trans_ivector, 2)

			common_vector = torch.matmul(torch.diag(num_examples/(num_examples*self.psi+1)), \
				                          -1*trans_ivector_sq*torch.pow(factor, 3)/self.dim)

			for i in range(self.dim):
				norm_drv[:,i] += common_vector
				norm_drv[i][i] += factor

			transform_drv = torch.matmul(self.transform.t(), norm_drv)
		else:
			transform_drv = self.transform.t()

		return transform_drv

	def DRV_Scores(self, trans_trainivector, num_examples, trans_testivector):
		mean = torch.zeros(self.dim)
		v1 = torch.zeros(self.dim)

		for i in range(self.dim):
			mean[i] = num_examples*self.psi[i]/(num_examples*self.psi[i]+1.0)*trans_trainivector[i]
			v1[i] = 1.0+self.psi[i]/(num_examples*self.psi[i]+1.0)

		v1 = 1.0/v1
		v2 = 1.0/(1+self.psi)

		score_drv = torch.matmul(torch.diag(trans_testivector), v2)-torch.matmul(torch.diag(trans_testivector-mean), v1)

		return score_drv
	
	def to(self, device):

		if device == self.device:
			return
		
		self.device = device
		self.mean = self.mean.to(self.device)
		self.transform = self.transform.to(self.device)
		self.psi = self.psi.to(self.device)