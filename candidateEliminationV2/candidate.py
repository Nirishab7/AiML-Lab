import csv

data=csv.reader(open("candidate/car.csv"))
a=[]
for x in data:
    a.append(tuple(x))

num_attr=len(a[0])-1

S=['0']*num_attr
G=['?']*num_attr
i=0
print('S[{0}]: '.format(i),S)
print('G[{0}]: '.format(i),G)
print("-----------")

for s in range(1,len(a)):
    if a[s][-1]=='YES':
        for j in range(num_attr):
            S[j]=a[s][j]
        break
temp=[]

for i in range(1,len(a)):
    if a[i][-1]=='YES':
        for j in range(num_attr):
            if S[j]!='?' and S[j]!=a[i][j]:
                S[j]='?'
        for j in range(num_attr):     #loops runs when len(temp) is greater than 1
            k=0
            l=len(temp)
            while k<l:
                if temp[k][j]!='?' and temp[k][j]!=S[j]:
                    del temp[k]
                    l-=1
                k+=1

    if a[i][-1]=='NO':
        if len(temp)==0:
            for j in range(num_attr):
                if S[j]!='?' and S[j]!=a[i][j]:
                    G[j]=S[j]
                    temp.append(G)
                    G=['?']*num_attr
        else:
            v='?'
            for h in range(len(temp)):
                c=0
                for j in range(num_attr):
                    if temp[h][j]!=a[i][j]:
                        c+=1
                        if temp[h][j]!='?':
                            v=temp[h][j]

                if c<num_attr:
                    G=temp[h].copy()
                    hypo=temp[h].copy()
                    del temp[h]
                    for j in range(num_attr):
                        if S[j]!='?' and S[j]!=a[i][j] and S[j]!=v:
                            G[j]=S[j]
                            temp.append(G)
                            G=hypo.copy()



    if i<s:
        print('S[{0}]: '.format(i),['0']*num_attr)
    else:
        print('S[{0}]: '.format(i),S)

    if len(temp)==0:
        print('G[{0}]: '.format(i),G)
    else:
        print('G[{0}]: '.format(i),temp)
    print("-----------")
