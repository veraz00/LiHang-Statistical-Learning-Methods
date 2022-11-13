import collections 

class Solution:
    def findLeastNumOfUniqueInts(self, arr, k):
        dict1 = collections.Counter(arr)
        dict2 = dict()
        for key, value in dict1.items():
            if value not in dict2.keys():
                dict2[value] = 1
            else:
                dict2[value] += 1
        print('dict2: ', dict2)
        for i in sorted(dict2.keys()):
            while dict2[i] > 0 and k > 0:
                k -= i
                dict2[i] -= 1
            if k == 0:
                break
        print(dict2, '---------')  
        return sum(dict2.values())

            

a = Solution()
s = [4,3,1,1,3,3,2]
print(a.findLeastNumOfUniqueInts(s, 3))