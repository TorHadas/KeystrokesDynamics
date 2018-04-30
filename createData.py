for i in range(5):
	for j in range(1, 32):
		for output in range(100, 110):
			with open(str(i)+"-"+str(j)+"--"+str(output)+".txt", "w") as text_file:	
				text_file.write(str(i)+"\n")
				for s in range((i)*5):
					text_file.write("0\n")
				num = format(j, '05b')

				for s in range(5):
					if len(num) <= s or num[s] != '1':
						text_file.write("0")
					else:
						text_file.write(str(output))
					text_file.write("\n")
				for s in range((4-i)*5):
					text_file.write("0\n")