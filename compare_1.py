import csv
import cv2
import numpy as np
import sklearn
import time

print('\n\t\t\t CHAPTER 1 \n')

y_log = []
output = []
gender_submission = []
with open('gender_submission.csv', 'r') as f:
	reader = csv.reader(f)
	next (reader)						#	Пропускакам первую строку в которой содержатся названия колонок
	for line in reader:
		gender_submission.append(line)
	# print('\n', gender_submission[:10])
	# print('\n\tNumber of lines on test log: ', len(gender_submission))



range = np.arange(892, 1310)

with open("output_own_3.csv", "r") as f:
    reader = csv.reader(f)
    next (reader)						#	Пропускакам первую строку в которой содержатся названия колонок
    for line in reader:
    	output.append(line)
    # print('\n', output[:10])
    # print('\n\tNumber of lines on test log: ', len(output))


count 		= 0
coincidence = 0 
overall		= 0

for f in output:
	# print('\n', count)
	# print('\n', f[1])
	# print('\n', gender_submission[count][1])
	if f[1] == '1' and gender_submission[count][1] == '1' :
		# print('Output: {} and Reality: {}'.format(f[1], gender_submission[count][1]))
		coincidence += 1

	if gender_submission[count][1] == '1':
		overall += 1
		
	count += 1 

	if count == 418:
		print('\n\t How many coincidences: ', coincidence)
		print('\n\t How many servivors in reality: ', overall)
		print('\n\t How many laps in total: ', count)
		acc    = (coincidence/overall)*100
		print('\n\t Accuracy: ', acc)




# print('\n\t\t\t CHAPTER. 2  \n')

# output_my = []
# with open("output_my.csv", "r") as f:
#     reader = csv.reader(f)
#     next (reader)						#	Пропускакам первую строку в которой содержатся названия колонок
#     for line in reader:
#     	output_my.append(line)
#     # print('\n', output_my[:10])
#     # print('\n\tNumber of lines on test log: ', len(output_my))

# count 		= 0
# coincidence = 0 
# overall		= 0

# for f in output_my:
# 	# print('\n', count)
# 	# print('\n', f[1])
# 	# print('\n', gender_submission[count][1])
# 	if f[1] == '1' and gender_submission[count][1] == '1' :
# 		# print('Output: {} and Reality: {}'.format(f[1], gender_submission[count][1]))
# 		coincidence += 1

# 	if gender_submission[count][1] == '1':
# 		overall += 1
		
# 	count += 1 

# 	if count == 418:
# 		print('\n\t How many coincidences: ', coincidence)
# 		print('\n\t How many servivors in reality: ', overall)
# 		print('\n\t How many laps in total: ', count)




