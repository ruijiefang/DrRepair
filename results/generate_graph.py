import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def cmp(a):
    temp = a[0].split(',')[1].strip()
    if "Clang" in temp:
        return "Z"
    return temp

if __name__ == "__main__":
    rows = []
    label = []
    first_row = True
    with open('output2.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if first_row:
                label = row
                first_row = False
            else:
                rows += [row]

    data = np.array(rows).astype(np.float)
    titles = []
    for x in label:
        if x.split(',')[0] not in titles and x != '':
            titles.append(x.split(',')[0])
    print(titles)

    for i in range(len(label)):
        label[i] = (label[i], i)
    label = label[1:]
        
    label.sort(key = cmp)
    # print(label)
    # for l in label:
    #     print(l)
    # exit(0)
    # print(titles)
    # print(len(data))
    # print(len(data[0]))
    # print(len(label))

    sz = 50
    ablations = [('code-only', 0, 150000), ('code-compiler--no-graph', 0, 150000), ('code-compiler--2l-graph', 0, 150000), ('code-compiler--2l-graph--pretrain', 0, 400000), ('code-compiler--2l-graph--finetune', 400000, 550000)]

    colors = cm.rainbow([0, 0.5, 1, 0.75])

    for t in titles:
    # t = 'pn_train_loss_localize'
        for ab in ablations:
#             s = '''
# \\begin{figure}[ht]
#   \\centering
#   \\includegraphics[width=8cm]{%s}
# \\end{figure}'''
#             print(s % f'graphs/{t[3:]}, {ab[0]}.jpg')
#             continue

            plt.figure()
            idx = 0
            for i in range(len(label)):
                if (t in label[i][0]) and (f'{ab[0]}/' in label[i][0]):
                    # area = np.pi*3
                    # plt.scatter(data[:,0], data[:,i], s=area, c=[colors[i]], alpha=1)
                    lab = label[i][0].split(',')[1].strip()[:-1]
                    # print(ab[2], data.shape, data[ab[1]//sz,0], ab[1]//sz)

                    if 'dev' in t:
                        space = 100
                    else:
                        space = 25
                    
                    # don't display incorrect 0s at beginning
                    start = ab[1]//sz+space-1
                    for j in range(ab[1]//sz+space-1, ab[2]//sz, space):
                        if data[j, label[i][1]] != 0:
                            start = j
                            break
                    # print(t, f'{ab[0]}/', label[i][0], start)
                    plt.plot(data[start:ab[2]//sz:space,0], data[start:ab[2]//sz:space,label[i][1]], color=colors[idx], label=lab)
                    idx += 1

            plt.title(f'{t[3:]}, {ab[0]}')
            if 'loss' in t:
                plt.ylabel('Loss')
            elif 'accuracy' in t:
                plt.ylabel('Accuracy')
            else:
                plt.ylabel('Norm')
            plt.xlabel('Iteration')
            plt.legend(title="Model", prop={'size': 8})
            # plt.legend(title="Ablation")
            plt.savefig(f'{t[3:]}, {ab[0]}.jpg')
            # plt.show()


    # plt.plot(data[0,:])
    # print(data[:,0])
    # print(data[:,1])
    # lab = label[1].split(',')[1].strip()
    # plt.plot(data[:1000,0], data[:1000,1], label=lab)

    # plt.title('pn_train_loss_localize')
    # plt.xlabel('Iteration')
    # plt.ylabel('pn_train_loss_localize')
    # plt.ylim(0.8, 1.02)
    # plt.xlim(0.8, 1.02)

    # plt.show()

    # for l in label:
    #     print(l)
    # print(label)
    # print(rows)
    # print(len(rows))
# for i in range(0, len(rows), 20):
#     print(rows[i])