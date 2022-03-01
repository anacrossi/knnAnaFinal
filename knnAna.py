
import argparse, operator
import numpy as np
import multiprocessing as mp
import time

def processEntrada (dataSet):

	atribs = []
	labels = []

	for line in dataSet:

		# separate the label
		labels.append(line[0])

		# parse the data, getting only the values 
		atribs.append([float(value.split(":")[1]) for value in line[1:]])

	return np.array(atribs), np.array(labels)

def knn (atributosTreino, labelsTreino, atributosDeTeste, k):

	# get the neigbors
	neigh = encontraVizinhos(atributosTreino, labelsTreino, atributosDeTeste, k) 

	# get the class of the vizinhos
	result = formaClasse(neigh)  

	return result

def encontraVizinhos (atributosTreino, labelsTreino, atributosDeTeste, k): #passa os atributos de treino, as labels de treino, os atributos do teste e o valor de k

	
	distancia = [(label, distEuclid(atributosDeTeste, x)) for x, label in zip(atributosTreino, labelsTreino)]  #monta vetor com a label do treino passa p funcao atributos de 
														                                                       #teste, atributos de treino e label                              
                                                                                                              #de treino
	distancia.sort(key = operator.itemgetter(1))                                   #ordena as diatncias

	vizinhos = []

	# get the k closest classes
	for i in range(k):
		vizinhos.append(distancia[i][0])                                              # k vizinhos de menor distancia

	return vizinhos

def formaClasse (vizinhos):                 #define a classificacao

	votes = {}
	
	for i in range(len(vizinhos)):
		classificado = vizinhos[i]

		if classificado in votes:
			votes[classificado] += 1
		else:
			votes[classificado] = 1

	# sort the number of votes, descending
	sortedVotes = sorted(votes.items(), key = operator.itemgetter(1), reverse = True)

	return sortedVotes[0][0]


def distEuclid (element1, element2): #calcula a distancia euclidiana dos vetores

	# subtract the arrays
	dif = (element1 - element2)         #um elemento - outro 
	# sum the results squared
	return np.sum(dif ** 2)              #soma todos os elementos e eleva ao quadrado



def calculaAcuracia (labelsTeste, classificacao):

	correct = 0
	length = len(labelsTeste)

	for i in range(length):

		# if predicted correctly
		if labelsTeste[i] == classificacao[i]:
			correct += 1

	# return accuracy
	return (correct/length)


def geraMatrizDeConfusao (labelsTeste, classificacao):

	# get the number of classes
	size = len(np.unique(labelsTeste))

	# create an empty matrix
	m = np.zeros([size, size], dtype = int)

	for i in range(len(labelsTeste)):

		# add one to each class predicted
		m[int(labelsTeste[i])][int(classificacao[i])] += 1

	return m




def main ():

	parser = argparse.ArgumentParser()
	parser.add_argument('train', type = argparse.FileType('r'))
	parser.add_argument('test', type = argparse.FileType('r'))
	parser.add_argument('k', type = int)
	args = parser.parse_args()

	arqTreino = getattr(args, 'train')
	arqTeste = getattr(args, 'test')
	kVizinhos = getattr(args, 'k')

	conjuntoTreino = []
	conjuntoTeste = []

	for line in arqTreino:
		conjuntoTreino.append(list(line.split())) #separa as linhas em varios elementos

	for line in arqTeste:
		conjuntoTeste.append(list(line.split()))

	atributosTreino, labelsTreino = processEntrada(conjuntoTreino) #realiza processamento de forma que devolve as labels como vetor y e os atributos como vetor x
	atributosTeste, labelsTeste = processEntrada(conjuntoTeste)		#realiza processamento de forma que devolve as labels como vetor y e os atributos como vetor x

	classific = []                                #classificacao

	pool = mp.Pool(mp.cpu_count())
	parametros = [(atributosTreino, labelsTreino, x, kVizinhos) for x in atributosTeste]
	classific = pool.starmap(knn, parametros)              #passa para o knn os labels de treino, os atributos de treino, varia os atributos e passa o valor de K
	pool.close()                                           #retorna a classificacao
	
	acuracia = calculaAcuracia(labelsTeste, np.array(classific))           #calula a acuracia
	print("Acurácia: ", acuracia)
	
	matrizDeConfusao = geraMatrizDeConfusao(labelsTeste, np.array(classific))  #monta matriz de confusao
	print("Matriz de Confusão")
	print(matrizDeConfusao)

# ------------------------------------------------------------------------

if __name__ == "__main__":
	main()

