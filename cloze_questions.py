from numpy.linalg import norm
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import ClozeQuestion
import numpy as np

FICHEIRO_PERGUNTAS = "resources/cloze_questions_REAP.txt"

GPT2_PT = 'pierreguillou/gpt2-small-portuguese'
BERT_BASE_PT = 'neuralmind/bert-base-portuguese-cased'
BERT_LARGE_PT = 'neuralmind/bert-large-portuguese-cased'
BERT_BASE_ML = 'bert-base-multilingual-cased'
model_name = BERT_BASE_PT
TOP_RESULTS = 5000

def report_fill_mask(perguntas, model, top=1000):
    fill = pipeline('fill-mask', model=model, tokenizer=model)

    certas_total = 0
    total_pos = {'N': 0, 'V': 0, 'A': 0, 'ADV': 0}
    certas_pos = {'N': 0, 'V': 0, 'A': 0, 'ADV': 0}

    por_pos = {
        'N': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0},
        'V': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0},
        'A': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0},
        'ADV': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0}
    }
    fora_top = 0
    rankings = []

    for p in perguntas:
        print(p)

        #total_pos[p.pos] = total_pos[p.pos] + 1
        por_pos[p.pos]['total'] += 1
        respondeu = False
        results = fill(p.pergunta, top_k=top)
        print('\t*', results[0]['token_str'])  # primeira resposta
        for i, r in enumerate(results):
            if r['token_str'] in p.alternativas:
                respondeu = True
                if r['token_str'] == p.certa:
                    certas_total += 1
                    #certas_pos[p.pos] = certas_pos[p.pos] + 1
                    por_pos[p.pos]['certas'] += 1
                    rankings.append(i)

                    if i < 10:
                        por_pos[p.pos]['top-10'] += 1
                        if i == 0:
                            por_pos[p.pos]['top-1'] += 1

                    print('\t+', i, r['token_str'])
                else:
                    print('\tx', i, r['token_str'])
                break

        if not respondeu:
            fora_top += 1
            por_pos[p.pos]['fora-top'] += 1

    print(certas_total / len(perguntas))
    print('Fora do top', fora_top)
    top_1 = rankings.count(0)
    print('Em #1:', top_1, top_1 / len(perguntas))
    top_10 = sum(r <= 10 for r in rankings)
    print('Top-10:', top_10, top_10 / len(perguntas))
    print(rankings)

    print('TOP (1, 10) -----')
    for k in por_pos:
        top_1_pos = por_pos[k]['top-1'] / por_pos[k]['total']
        top_10_pos = por_pos[k]['top-10'] / por_pos[k]['total']
        print('\t', k, str(top_1_pos), str(top_10_pos))

    print('CERTAS -----')
    for k in por_pos:
        prop_certas = por_pos[k]['certas'] / por_pos[k]['total']
        print('\t', k, str(prop_certas))

    print('FORA do TOP -----')
    for k in por_pos:
        prop_fora = por_pos[k]['fora-top'] / por_pos[k]['total']
        print('\t', k, str(prop_fora))


def report_feature_extraction(perguntas, model, embedding='first'):

    #features = pipeline('feature-extraction', model=model)
    features = pipeline('feature-extraction', model=model, tokenizer=model)

    certas_total = 0
    total_pos = {'N': 0, 'V': 0, 'A': 0, 'ADV': 0}
    certas_pos = {'N': 0, 'V': 0, 'A': 0, 'ADV': 0}
    for p in perguntas:
        print(p)
        total_pos[p.pos] = total_pos[p.pos] + 1

        features_p = features(p.pergunta)[0][0] if type == 'first' else mean_embedding(features(p.pergunta)[0])
        features_alt = [features(a)[0][0] for a in p.alternativas]
        max_sim = -1
        best_i = -1
        for i, fa in enumerate(features_alt):
            sim = cosine(features_p, fa)
            if sim > max_sim:
                max_sim = sim
                best_i = i
        if p.alternativas[best_i] == p.certa:
            print('\t+', p.alternativas[best_i])
            certas_total += 1
            certas_pos[p.pos] = certas_pos[p.pos] + 1
        else:
            print('\t-', p.alternativas[best_i])

    print(certas_total / len(perguntas))

    for k in total_pos:
        prop_certas = certas_pos[k] / total_pos[k]
        print('\t', k, str(prop_certas))


def mean_embedding(vector):
    return np.mean(vector, axis=0)

def cosine(v1, v2):
    if all(v == 0 for v in v1) or all(v == 0 for v in v2):
        return 0.0
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def report_fill_mask_gpt(perguntas, model, top=1000):
    generate = pipeline('text-generation', model=model)
    tokenizer = GPT2Tokenizer.from_pretrained(model)

    certas_total = 0
    por_pos = {
        'N': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0},
        'V': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0},
        'A': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0},
        'ADV': {'total': 0, 'certas': 0, 'top-1': 0, 'top-10': 0, 'fora-top': 0}
    }
    fora_top = 0
    rankings = []

    for p in perguntas:
        print(p, flush=True)
        idx = p.pergunta.index('[MASK]')
        pp = p.pergunta[:idx].strip()

        #total_pos[p.pos] = total_pos[p.pos] + 1
        por_pos[p.pos]['total'] += 1
        respondeu = False

        n_tokens = len(tokenizer(pp)['input_ids'])
        #print('#tokens =', n_tokens)
        results = generate(pp, max_length=n_tokens+1, pad_token_id=50256, temperature=1.0, num_return_sequences=top)
        #top_p=0.9,

        primeira = results[0]['generated_text'].strip()
        print('\t*', primeira[idx:])  # primeira resposta
        for i, r in enumerate(results):
            resposta = results[i]['generated_text'][idx:].strip()

            if resposta in p.alternativas:
                respondeu = True
                if resposta == p.certa:
                    certas_total += 1
                    #certas_pos[p.pos] = certas_pos[p.pos] + 1
                    por_pos[p.pos]['certas'] += 1
                    rankings.append(i)

                    if i < 10:
                        por_pos[p.pos]['top-10'] += 1
                        if i == 0:
                            por_pos[p.pos]['top-1'] += 1

                    print('\t+', i, resposta)
                else:
                    print('\tx', i, resposta)
                break

        if not respondeu:
            fora_top += 1
            por_pos[p.pos]['fora-top'] += 1

    print('Total certas:', certas_total / len(perguntas))
    print('Fora do top', fora_top)
    top_1 = rankings.count(0)
    print('Em #1:', top_1, top_1 / len(perguntas))
    top_10 = sum(r <= 10 for r in rankings)
    print('Top-10:', top_10, top_10 / len(perguntas))
    print(rankings)

    print('TOP (1, 10) -----')
    for k in por_pos:
        top_1_pos = por_pos[k]['top-1'] / por_pos[k]['total']
        top_10_pos = por_pos[k]['top-10'] / por_pos[k]['total']
        print('\t', k, str(top_1_pos), str(top_10_pos))

    print('CERTAS -----')
    for k in por_pos:
        prop_certas = por_pos[k]['certas'] / por_pos[k]['total']
        print('\t', k, str(prop_certas))

    print('FORA do TOP -----')
    for k in por_pos:
        prop_fora = por_pos[k]['fora-top'] / por_pos[k]['total']
        print('\t', k, str(prop_fora))
    print(flush=True)


def report_most_probable_gpt(perguntas, model_name):
    certas_total = 0
    total_pos = {'N': 0, 'V': 0, 'A': 0, 'ADV': 0}
    certas_pos = {'N': 0, 'V': 0, 'A': 0, 'ADV': 0}

    print('Load', model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    for p in perguntas:
        print(p, flush=True)
        total_pos[p.pos] = total_pos[p.pos] + 1

        options = [p.pergunta.replace('[MASK]', a) for a in p.alternativas]
        #options.append(p.certa)
        melhor = most_probable_gpt(model, tokenizer, options)
        resposta = p.alternativas[options.index(melhor)]

        if resposta == p.certa:
            print('\t+', resposta)
            certas_total += 1
            certas_pos[p.pos] = certas_pos[p.pos] + 1
        else:
            print('\t-', resposta)

    print('Certas', certas_total / len(perguntas))

    for k in total_pos:
        prop_certas = certas_pos[k] / total_pos[k]
        print('\t', k, str(prop_certas))


#https://stackoverflow.com/questions/63543006/how-can-i-find-the-probability-of-a-sentence-using-gpt-2
def most_probable_gpt(model, tokenizer, frases):
    frases_score = []
    for frase in frases:
        #input_ids = token indices, numerical representations of tokens building the sequences that will be used as input by the model.
        tokens_tensor = tokenizer.encode(frase, add_special_tokens=False, return_tensors="pt")
        #predictions
        loss = model(tokens_tensor, labels=tokens_tensor)[0]
        frases_score.append((frase, np.exp(loss.cpu().detach().numpy())))
    frases_score.sort(key=lambda tup: tup[1])
    return frases_score[0][0]


if __name__ == '__main__':
    perguntas = ClozeQuestion.carrega_perguntas(FICHEIRO_PERGUNTAS)

    report_most_probable_gpt(perguntas, GPT2_PT)
    #feature_extraction_report(perguntas, GPT2_PT, embedding='mean')
    #fill_mask_report(perguntas, model_name, top=TOP_RESULTS)
    #fill_mask_gpt_report(perguntas, GPT2_PT, top=1000)
    #feature_extraction_report(perguntas, model_name, embedding='first')
