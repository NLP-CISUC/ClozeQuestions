import re
import random

def carrega_perguntas(ficheiro, top=-1, distratores=2):
    perguntas = []
    with open(ficheiro) as f:
        state = 0
        for row in f:
            conteudo = row.strip()

            #print(state, row)

            if state == 0:
                if len(conteudo) > 0 and row[0] != '#':
                    id = conteudo
                    state = 1
            elif state == 1:
                #lema resposta
                state = 2
            elif state == 2:
                resposta = conteudo
                state = 3
            elif state == 3:
                pos = conteudo.split(':')[0]
                state = 4
            elif state == 4:
                #dificuldade
                state = 5
            elif state == 5:
                frase = conteudo.replace('____', '[MASK]')
                state = 6
            elif state == 6:
                if distratores == 1:
                    alternativas = re.sub(r'\([^)]*\)', '', conteudo).split(' ')
                state = 7
            elif state == 7:
                if distratores == 2:
                    alternativas = re.sub(r'\([^)]*\)', '', conteudo).split(' ')
                #print('P', id, frase, pos, resposta, distratores)
                alternativas.append(resposta)
                random.shuffle(alternativas)
                perg = ClozeQuestion(id, frase, pos, resposta, alternativas)
                perguntas.append(perg)

                if top > 0 and len(perguntas) == top:
                    break

                state = 0

    return perguntas


class ClozeQuestion:

    def __init__(self, id, pergunta, pos, certa, alternativas):
        self.id = id
        self.pergunta = pergunta
        self.pos = pos
        self.certa = certa
        self.alternativas = alternativas

    def __str__(self):
        return self.id + ": " + self.pergunta + "\t(" + self.pos +"), " + str(self.alternativas) + ", Certa=" + self.certa
