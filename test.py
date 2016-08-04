from numpy import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import survey
import Pmf

def toNdArray(table):
    records = table.records
    list = zeros((len(records), 5))
    for i in range(len(records)):
        list[i, 0] = records[i].caseid
        list[i, 1] = records[i].prglength
        list[i, 2] = records[i].outcome
        if records[i].birthord != 'NA':
            list[i, 3] = records[i].birthord
        else:
            list[i, 3] = -1
        list[i, 4] = records[i].finalwgt
    return list

def draw(fdata, odata, fpmf, opmf):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    color_index = ['r', 'g', 'b']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    ax1.bar(fdata[:, 0] + 0*.25 + .2, fdata[:, 1], width = .25, color = color_index[0], alpha = .5, label='first')
    ax1.bar(odata[:, 0] + 1*.25 + .5, odata[:, 1], width = .25, color = color_index[1], alpha = .5, label='other')
    #ax.set_ylim([0, 0.8])
    ax1.set_xlim([20, 50])
    ax1.set_xlabel('week')
    ax1.set_ylabel('probability')
    ax1.legend(loc=2)

    x = range(35, 46)
    y = []
    for i in x:
        y.append(100 * (fpmf.Prob(i) - opmf.Prob(i)))
    ax2.bar(x, y)
    ax2.set_xlabel('week')
    ax2.set_ylabel('first - other')

    plt.show()
    #plt.savefig('complex_bar_chart')

def getDataList():
    table = survey.Pregnancies()
    table.ReadRecords()
    list = toNdArray(table)
    return list

def partition(list):
    live = list[list[:, 2] == 1, :]
    first = live[live[:, 3] == 1, :]
    other = live[live[:, 3] != 1, :]
    return first[:, 1].tolist(), other[:, 1].tolist()

def getData(list):
    pmf = Pmf.MakePmfFromList(list)
    data = array(pmf.Items())
    return data, pmf

def probEarly(pmf):
    min, max = minmax(pmf)
    return probP(pmf, int(min), 37)

def probOnTime(pmf):
    return probP(pmf, 38, 40)

def probLate(pmf):
    min, max = minmax(pmf)
    return probP(pmf, 41, int(max))

def probP(pmf, begin, end):
    sum = 0
    for i in range(begin, end + 1):
        sum += pmf.Prob(i)
    return sum

def minmax(pmf):
    n = len(pmf.Items())
    min = pmf.Items()[0][0]
    max = pmf.Items()[n - 1][0]
    return min, max

list = getDataList()
first, other = partition(list)

fdata, fpmf = getData(first)
odata, opmf = getData(other)



#print probP(fpmf, 0, 37)
#draw(fdata, odata, fpmf, opmf)

print probEarly(fpmf)
print probEarly(opmf)




