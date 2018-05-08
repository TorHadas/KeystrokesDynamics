class Debug():
	def __init__(self):
		self.mode = True

	def off(self):
		self.mode = False

	def on(self):
		self.mode = True

	def log(self, str):
		if(self.mode):
			print(str)
