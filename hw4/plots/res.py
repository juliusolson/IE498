from matplotlib import pyplot as plt
import numpy as np
from sys import argv as argv


cifar100 = [
	{'Test accuracy': 0.1941257911392405, 'Train accuracy': 0.1834638746803069},
	{'Test accuracy': 0.28085443037974683, 'Train accuracy': 0.2758951406649616},
	{'Test accuracy': 0.3639240506329114, 'Train accuracy': 0.36345108695652173},
	{'Test accuracy': 0.41455696202531644, 'Train accuracy': 0.44093670076726343},
	{'Test accuracy': 0.44551028481012656, 'Train accuracy': 0.47642263427109977},
	{'Test accuracy': 0.4762658227848101, 'Train accuracy': 0.5243366368286445},
	{'Test accuracy': 0.5044501582278481, 'Train accuracy': 0.5548673273657289},
	{'Test accuracy': 0.5169106012658228, 'Train accuracy': 0.5846187659846548},
	{'Test accuracy': 0.5463805379746836, 'Train accuracy': 0.616008631713555},
	{'Test accuracy': 0.5479628164556962, 'Train accuracy': 0.6350103900255755},
	{'Test accuracy': 0.5661590189873418, 'Train accuracy': 0.6532328964194374},
	{'Test accuracy': 0.5711036392405063, 'Train accuracy': 0.6819453324808185},
	{'Test accuracy': 0.5743670886075949, 'Train accuracy': 0.6906170076726342},
	{'Test accuracy': 0.5863330696202531, 'Train accuracy': 0.7105778452685422},
	{'Test accuracy': 0.592068829113924, 'Train accuracy': 0.7297993925831202},
	{'Test accuracy': 0.5911787974683544, 'Train accuracy': 0.7415281329923273},
	{'Test accuracy': 0.6048259493670886, 'Train accuracy': 0.7567934782608695},
	{'Test accuracy': 0.6160996835443038, 'Train accuracy': 0.7929787404092071},
	{'Test accuracy': 0.6084849683544303, 'Train accuracy': 0.8004515664961637},
	{'Test accuracy': 0.6183742088607594, 'Train accuracy': 0.8150775255754475},
	{'Test accuracy': 0.6194620253164557, 'Train accuracy': 0.8236093350383632},
	{'Test accuracy': 0.6074960443037974, 'Train accuracy': 0.8288842710997443},
	{'Test accuracy': 0.6229232594936709, 'Train accuracy': 0.8430306905370843},
	{'Test accuracy': 0.6246044303797469, 'Train accuracy': 0.8531010230179028},
	{'Test accuracy': 0.6160996835443038, 'Train accuracy': 0.8585757672634271},
	{'Test accuracy': 0.625692246835443, 'Train accuracy': 0.8673273657289002},
	{'Test accuracy': 0.6251977848101266, 'Train accuracy': 0.873821131713555},
	{'Test accuracy': 0.6209454113924051, 'Train accuracy': 0.8730418797953964},
	{'Test accuracy': 0.6114517405063291, 'Train accuracy': 0.8858296035805626},
	{'Test accuracy': 0.619560917721519, 'Train accuracy': 0.8944213554987213}
]

tinyimage = [
	{"Train accuracy": 0.04490688938618926, "Test accuracy": 0.04430379746835443},
	{"Train accuracy" : 0.1432724584398977, "Test accuracy": 0.1317246835443038},
	{"Train accuracy" : 0.17508192135549872, "Test accuracy": 0.1607001582278481},
	{"Train accuracy" : 0.2626178868286445, "Test accuracy": 0.24208860759493672},
	{"Train accuracy" : 0.2928089034526854, "Test accuracy": 0.26305379746835444},
	{"Train accuracy" : 0.2693014705882353, "Test accuracy": 0.24386867088607594},
	{"Train accuracy" : 0.3551090952685422, "Test accuracy": 0.3047863924050633},
	{"Train accuracy" : 0.382872442455243, "Test accuracy": 0.3316851265822785},
	{"Train accuracy" : 0.4164601982097187, "Test accuracy": 0.3490901898734177},
	{"Train accuracy" : 0.3937619884910486, "Test accuracy": 0.32832278481012656},
	{"Train accuracy" : 0.460258152173913, "Test accuracy": 0.3692642405063291},
	{"Train accuracy" : 0.41844828964194375, "Test accuracy": 0.33277294303797467},
	{"Train accuracy" : 0.4858136189258312, "Test accuracy": 0.3822191455696203},
	{"Train accuracy" : 0.5076326726342711, "Test accuracy": 0.38686708860759494},
	{"Train accuracy" : 0.4718370364450128, "Test accuracy": 0.35986946202531644},
	{"Train accuracy" : 0.5362052429667519, "Test accuracy": 0.39566851265822783},
	{"Train accuracy" : 0.536894581202046, "Test accuracy": 0.3959651898734177},
	{"Train accuracy" : 0.5899736253196931, "Test accuracy": 0.42503955696202533},
	{"Train accuracy" : 0.58431905370844, "Test accuracy": 0.4107001582278481},
	{"Train accuracy" : 0.5621503356777494, "Test accuracy": 0.4017998417721519},
	{"Train accuracy" : 0.5336077365728901, "Test accuracy": 0.37005537974683544},
	{"Train accuracy" : 0.6422534367007673, "Test accuracy": 0.43225870253164556},
	{"Train accuracy" : 0.584259111253197, "Test accuracy": 0.4017009493670886},
	{"Train accuracy" : 0.6189358216112532, "Test accuracy": 0.41030458860759494},
	{"Train accuracy" : 0.6890085517902813, "Test accuracy": 0.4452136075949367},
	{"Train accuracy" : 0.6651015025575447, "Test accuracy": 0.4267207278481013},
	{"Train accuracy" : 0.6696671195652174, "Test accuracy": 0.4233583860759494},
	{"Train accuracy" : 0.6319333439897699, "Test accuracy": 0.39883306962025317},
	{"Train accuracy" : 0.666110533887468, "Test accuracy": 0.41060126582278483},
	{"Train accuracy" : 0.7414781809462916, "Test accuracy": 0.44452136075949367},
]


cifar100SGD = [
	{"Train accuracy": 0.18304427749360613, "Test accuracy" : 0.18888449367088608},
{"Train accuracy" : 0.18422314578005114, "Test accuracy" : 0.1885878164556962},
{"Train accuracy" : 0.3252677429667519, "Test accuracy" : 0.31536787974683544},
{"Train accuracy" : 0.32520780051150894, "Test accuracy" : 0.31774129746835444},
{"Train accuracy" : 0.4023937020460358, "Test accuracy" : 0.3752966772151899},
{"Train accuracy" : 0.4007552749360614, "Test accuracy" : 0.37153876582278483},
{"Train accuracy" : 0.4733655690537084, "Test accuracy" : 0.4345332278481013},
{"Train accuracy" : 0.47598305626598464, "Test accuracy" : 0.4321598101265823},
{"Train accuracy" : 0.5508711636828645, "Test accuracy" : 0.479628164556962},
{"Train accuracy" : 0.5504315856777494, "Test accuracy" : 0.48002373417721517},
{"Train accuracy" : 0.595548273657289, "Test accuracy" : 0.5129549050632911},
{"Train accuracy" : 0.5950887148337596, "Test accuracy" : 0.5134493670886076},
{"Train accuracy" : 0.6109135230179028, "Test accuracy" : 0.5130537974683544},
{"Train accuracy" : 0.6094749040920716, "Test accuracy" : 0.5151305379746836},
{"Train accuracy" : 0.6469389386189258, "Test accuracy" : 0.5347112341772152},
{"Train accuracy" : 0.6458799552429667, "Test accuracy" : 0.5350079113924051},
{"Train accuracy" : 0.6814458120204604, "Test accuracy" : 0.546875},
{"Train accuracy" : 0.6823249680306905, "Test accuracy" : 0.549248417721519},
{"Train accuracy" : 0.7147338554987213, "Test accuracy" : 0.5578520569620253},
{"Train accuracy" : 0.7153132992327366, "Test accuracy" : 0.5545886075949367},
{"Train accuracy" : 0.7371323529411765, "Test accuracy" : 0.5692246835443038},
{"Train accuracy" : 0.734315057544757, "Test accuracy" : 0.5658623417721519},
{"Train accuracy" : 0.768781969309463, "Test accuracy" : 0.5727848101265823},
{"Train accuracy" : 0.7689617966751918, "Test accuracy" : 0.5769382911392406},
{"Train accuracy" : 0.7776934143222506, "Test accuracy" : 0.5687302215189873},
{"Train accuracy" : 0.779511668797954, "Test accuracy" : 0.5692246835443038},
{"Train accuracy" : 0.8094828964194374, "Test accuracy" : 0.5745648734177216},
{"Train accuracy" : 0.8091831841432225, "Test accuracy" : 0.5744659810126582},
{"Train accuracy" : 0.8253276854219949, "Test accuracy" : 0.5902887658227848},
{"Train accuracy" : 0.8225503516624041, "Test accuracy" : 0.5914754746835443},
{"Train accuracy" : 0.81559702685422, "Test accuracy" : 0.5681368670886076},
{"Train accuracy" : 0.8124000959079284, "Test accuracy" : 0.5673457278481012},
{"Train accuracy" : 0.8384550831202046, "Test accuracy" : 0.5747626582278481},
{"Train accuracy" : 0.8392143542199488, "Test accuracy" : 0.575059335443038},
{"Train accuracy" : 0.8558983375959079, "Test accuracy" : 0.5933544303797469},
{"Train accuracy" : 0.8570971867007673, "Test accuracy" : 0.5922666139240507},
{"Train accuracy" : 0.8754995204603581, "Test accuracy" : 0.5833662974683544},
{"Train accuracy" : 0.8780770460358056, "Test accuracy" : 0.5805973101265823},
{"Train accuracy" : 0.8802949168797954, "Test accuracy" : 0.5878164556962026},
{"Train accuracy" : 0.8776974104859335, "Test accuracy" : 0.5841574367088608},
{"Train accuracy" : 0.893602141943734, "Test accuracy" : 0.5908821202531646},
{"Train accuracy" : 0.8921835038363172, "Test accuracy" : 0.5902887658227848},
{"Train accuracy" : 0.8972985933503836, "Test accuracy" : 0.5846518987341772},
{"Train accuracy" : 0.8985773657289002, "Test accuracy" : 0.5850474683544303},
{"Train accuracy" : 0.9087076406649617, "Test accuracy" : 0.5936511075949367},
{"Train accuracy" : 0.90934702685422, "Test accuracy" : 0.5949367088607594},
{"Train accuracy" : 0.9137228260869565, "Test accuracy" : 0.5908821202531646},
{"Train accuracy" : 0.9134630754475703, "Test accuracy" : 0.5921677215189873},
{"Train accuracy" : 0.9106257992327366, "Test accuracy" : 0.5905854430379747},
{"Train accuracy" : 0.9102261828644501, "Test accuracy" : 0.5915743670886076},
{"Train accuracy" : 0.9354819373401535, "Test accuracy" : 0.5975079113924051},
{"Train accuracy" : 0.9364010549872123, "Test accuracy" : 0.5935522151898734},
{"Train accuracy" : 0.9198769181585678, "Test accuracy" : 0.5992879746835443},
{"Train accuracy" : 0.9210557864450127, "Test accuracy" : 0.5999802215189873},
{"Train accuracy" : 0.924852141943734, "Test accuracy" : 0.5936511075949367},
{"Train accuracy" : 0.9260709718670077, "Test accuracy" : 0.5922666139240507},
{"Train accuracy" : 0.923233695652174, "Test accuracy" : 0.5765427215189873},
{"Train accuracy" : 0.92349344629156, "Test accuracy" : 0.5752571202531646},
{"Train accuracy" : 0.9470108695652174, "Test accuracy" : 0.6050237341772152},
{"Train accuracy" : 0.947170716112532, "Test accuracy" : 0.6006724683544303},
]

pretrained = [
	{"Train accuracy": 0.5304907289002557, "Test accuracy": 0.627373417721519},
	{"Train accuracy": 0.5973665281329923, "Test accuracy": 0.6874011075949367},
	{"Train accuracy": 0.6282568734015346, "Test accuracy": 0.7100474683544303},
	{"Train accuracy": 0.6559702685421995, "Test accuracy": 0.7378362341772152},
	{"Train accuracy": 0.6855618606138107, "Test accuracy": 0.7409018987341772},
	{"Train accuracy": 0.6952325767263428, "Test accuracy": 0.7509889240506329},
	{"Train accuracy": 0.7158727621483376, "Test accuracy": 0.7604825949367089},
	{"Train accuracy": 0.7192495204603581, "Test accuracy": 0.7613726265822784},
	{"Train accuracy": 0.7300791240409207, "Test accuracy": 0.7569224683544303},
	{"Train accuracy": 0.735673753196931, "Test accuracy": 0.760185917721519}
]

datasets = {
	"C100": cifar100, 
	"TINY": tinyimage,
	"SGD": cifar100SGD,
	"PRE": pretrained
}


name = argv[1]
data = datasets[name]
x = np.array([i for i in range(len(data))])
y_train = np.array([x["Train accuracy"] for x in data])
y_test = np.array([x["Test accuracy"] for x in data])


plt.plot(x, y_train, linewidth=1.0, label="Train Accuracy")
plt.plot(x, y_test, linewidth=1.0, label="Test Accuracy")
plt.legend()
plt.title("Train and Test accuracy")
#plt.show()
plt.savefig(f"{name}.png", dpi=400)