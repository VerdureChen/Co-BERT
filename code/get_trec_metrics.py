import subprocess
import gc


def run(command, get_ouput=False):
  try:
    if get_ouput:
      process = subprocess.Popen(command, stdout=subprocess.PIPE)
      output, err = process.communicate()
      exit_code = process.wait()
      return output
    else:
      subprocess.call(command)
  except subprocess.CalledProcessError as e:
    print(e)


def trec_eval(qrelf, runf, metric1, metric2, metric3, metric4, trec_eval_f):
    command = [trec_eval_f, '-m', metric1, '-m', metric2, '-m', metric3,'-m', metric4, qrelf, runf]
    output = run(command, get_ouput=True)
    output = str(output, encoding='utf-8')
    output = output.replace('\t', ' ').split('\n')
    # assert len(output) == 1
    # print(output)
    score_dict={}
    for item in output:
        if item is not '':
            # print(item.split(' '))
            score_dict[item.split(' ')[0]] = item.split(' ')[-1]
    return score_dict

def validate(qrelf, runf, trec_eval_f):
    VALIDATION_METRIC1 = 'P.20'
    VALIDATION_METRIC2 = 'map_cut.100'
    VALIDATION_METRIC3 = 'map'
    VALIDATION_METRIC4 = 'ndcg_cut.20'
    return trec_eval(qrelf, runf, VALIDATION_METRIC1, VALIDATION_METRIC2,
                     VALIDATION_METRIC3, VALIDATION_METRIC4, trec_eval_f)


if __name__ == '__main__':
    qrelf = r''
    runf = r''
    for i in range(1,2):
        score = validate(qrelf,runf.format(str(i)))
        # print(f'fold {i}')
        for key, val in score.items():
            print(f"{i}:{key}\t{val}")
        print('\n')