#abstract class defining the base graph learning model functionality

from abc import ABC, abstractmethod

class BaseTrainer(ABC):

	@abstractmethod
	def __init__(self, args):
		pass
	
	@abstractmethod
	def build_model(self):
		pass
		
	@abstractmethod
	def train(self):
		pass
	
	@abstractmethod
	def evaluate(self):
		pass