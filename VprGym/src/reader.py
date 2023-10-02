# Fetching WL and CPD information from log file
WLHeader = 'BB estimate of min-dist (placement) wire length: '
CPDHeader = 'Placement estimated critical path delay (least slack): '
RTHeader = '# Placement took '
ZMQTHeader = 'Zeromq elapsed time (seconds): '
SWAPHeader = 'Swaps called:'


def read_WL_CPD():
	WL = None
	CPD = None
	PT = None
	ZMQT = None
	SWAP = None
	f = open('vpr_stdout.log')
	lines = f.readlines()
	for line in lines:
		if WLHeader in line:
			WL = int(line.split()[-1])
		if CPDHeader in line:
			CPD = float(line.split()[7])
		if RTHeader in line:
			PT = float(line.split()[3])
		if ZMQTHeader in line:
			ZMQT = float(line.split()[-1])
		if SWAPHeader in line:
			SWAP = int(line.split()[-1])
	return WL, CPD, PT, SWAP
