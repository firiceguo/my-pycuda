import matplotlib.pyplot as plt


def getData(file):
    x = []
    time = []
    f = open(file, 'r')
    while 1:
        line = f.readline()
        if line:
            i = 0
            temp = ''
            while not line[i].isdigit():
                i += 1
            while line[i].isdigit():
                temp += line[i]
                i += 1
            x.append(int(temp))
            temp = ''
            while not line[i].isdigit():
                i += 1
            while line[i].isdigit() or line[i] == '.':
                temp += line[i]
                i += 1
            time.append(float(temp))
        else:
            break
    return(x, time)

if __name__ == '__main__':
    radius = 10
    ReddFile = "../log/Redundant.log"
    FilterFile = "../log/filter.log"
    NaiveFile = "../log/naive.log"
    # SerialFile = "../log/serial.log"

    ReddX, ReddTime = getData(ReddFile)
    FilterX, FilterTime = getData(FilterFile)
    NaiveX, NaiveTime = getData(NaiveFile)
    # SerialX, SetialTime = getData(SerialFile)

    # plt.plot(SerialX, SetialTime, 'k', label="Setial Time")
    plt.plot(NaiveX, NaiveTime, 'r', label="Naive")
    plt.plot(FilterX, FilterTime, 'g', label="Separable Filter")
    plt.plot(ReddX, ReddTime, 'b', label="Redundant Boundary")
    plt.xlabel("Image radius")
    plt.ylabel("Time(ms)")
    plt.title("Time usage(Kernel Radius = " + str(radius) + ")")
    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.show()
    plt.savefig("out-r" + str(radius) + ".png")
    plt.close("all")
