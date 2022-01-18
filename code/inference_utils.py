import os
import logging
import argparse
from get_trec_metrics import validate
import shutil
import datetime
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_metrics_for_test(result_path, ref_file, output_path, ckpt_outpath, trec_eval):
    if not os.path.exists(os.path.join(ckpt_outpath, 'best4test')):
        os.makedirs(os.path.join(ckpt_outpath, 'best4test'))
    best_dir = os.path.join(ckpt_outpath, 'best4test')
    dev_record_file = os.path.join(output_path, 'record_{}.txt'.format(datetime.date.today().strftime('%Y_%m_%d')))
    max_score=-1
    max_outfile = ''
    total_metrics = {}
    with open(dev_record_file, 'w', encoding='utf-8') as dev_record:
        dir_list = [item for item in os.listdir(result_path) if item.startswith('results')]

        def num(ele):
            return int(ele.split('-')[-1].split('_')[0])
        try:
            dir_list.sort(key=num, reverse=True)
        except:
            pass
        logger.info('*******')
        logger.info(dir_list)
        logger.info('*******')
        if len(dir_list)==1:
            res_file = os.path.join(result_path, dir_list[0])
            path_to_candidate = res_file
            path_to_reference = ref_file
            metrics = validate(path_to_reference, path_to_candidate, trec_eval)
            dev_record.write('##########{}###########\n'.format(dir_list[0]))
            for metric in sorted(metrics):
                dev_record.write('{}: {}\n'.format(metric, metrics[metric]))
            dev_record.write('#####################\n')
            return metrics
        else:
            for i,fil in enumerate(dir_list):
                res_file = os.path.join(result_path, fil)
                path_to_candidate = res_file
                path_to_reference = ref_file
                metrics = validate(path_to_reference, path_to_candidate, trec_eval)
                total_metrics[10-i] = metrics
                # print(metrics)
                p20 = float(metrics['ndcg_cut_20'])
                if p20>max_score:
                    max_score = p20
                    max_outfile = fil
                dev_record.write('##########{}###########\n'.format(fil))
                for metric in sorted(metrics):
                    dev_record.write('{}: {}\n'.format(metric, metrics[metric]))
                dev_record.write('#####################\n')
            dev_record.write('MAX FILE:{}, MAX ndcg_cut_20:{}'.format(max_outfile, str(max_score)))
            check_name = max_outfile.split('_')[1]
            check_path = os.path.join(ckpt_outpath, check_name)
            best_dir = os.path.join(best_dir, check_name)
            shutil.copytree(check_path, best_dir)

            return total_metrics



def get_total_score(result_fold, output_path):
    folds = range(1, 6)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path,
                               'test_score_{}.txt'.format(result_fold.split('/')[-1]))
    with open(output_file, 'w', encoding='utf-8') as outf:
        count=0
        for fold in folds:
            total_metrics = {}
            result_path = result_fold.format(str(fold))
            dir_list = [item for item in os.listdir(result_path) if item.startswith('results')]

            def num(ele):
                return int(ele.split('-')[-1].split('_')[0])

            try:
                dir_list.sort(key=num)
            except:
                pass
            logger.info('*******')
            logger.info(dir_list)
            logger.info('*******')
            fil = dir_list[-1]
            res_file = os.path.join(result_path, fil)
            with open(res_file, 'r', encoding='utf-8') as res:
                print(fil)
                for line in res:
                    outf.write(line)
                    count+=1
    print(f'total_line:{str(count)}')


def get_metrics(ref_file, output_path, trec_eval):
    dev_record_file = os.path.join(output_path, 'record.txt')
    max_score=-1
    max_outfile = ''
    total_metrics = {}
    with open(dev_record_file, 'w', encoding='utf-8') as dev_record:
        dir_list = [item for item in os.listdir(output_path) if item.startswith('test_score')]

        def num(ele):
            return int(ele.split('-')[-1].split('_')[0])
        try:
            dir_list.sort(key=num, reverse=True)
        except:
            pass
        logger.info('*******')
        logger.info(dir_list)
        logger.info('*******')
        if len(dir_list)==1:
            res_file = os.path.join(output_path, dir_list[0])
            path_to_candidate = res_file
            path_to_reference = ref_file
            metrics = validate(path_to_reference, path_to_candidate, trec_eval)
            dev_record.write('##########{}###########\n'.format(dir_list[0]))
            for metric in sorted(metrics):
                dev_record.write('{}: {}\n'.format(metric, metrics[metric]))
            dev_record.write('#####################\n')
            return metrics
        else:
            for i,fil in enumerate(dir_list):
                res_file = os.path.join(output_path, fil)
                path_to_candidate = res_file
                path_to_reference = ref_file
                metrics = validate(path_to_reference, path_to_candidate, trec_eval)
                total_metrics[10-i] = metrics
                p20 = float(metrics['ndcg_cut_20'])
                if p20>max_score:
                    max_score = p20
                    max_outfile = fil
                dev_record.write('##########{}###########\n'.format(fil))
                for metric in sorted(metrics):
                    dev_record.write('{}: {}\n'.format(metric, metrics[metric]))
                dev_record.write('#####################\n')
            dev_record.write('MAX FILE:{}, MAX ndcg_cut_20:{}'.format(max_outfile, str(max_score)))
            return total_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path",
                        default=None,
                        type=str,
                        required=True,
                        help="result_path_without_fold.")
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        help="total_file_path.")
    parser.add_argument("--ref_file",
                        default=None,
                        type=str,
                        required=True,
                        help="qrel_file_path.")
    parser.add_argument("--ckpt_path",
                        default=None,
                        type=str,
                        help="data name")
    parser.add_argument("--trec_eval",
                        default=None,
                        type=str,
                        required=True,
                        help="trec_eval path.")
    parser.add_argument("--best_for_test",
                        action='store_true',
                        help="Whether to get the best dev ckpt.")
    args = parser.parse_args()
    logger.info('The args: {}'.format(args))
    # result_fold = '/data/chenxiaoyang/MS/BERT-QE/BERT-QE/gov2/output/m-10_base_base_base/fold-{}/rerank-1000_kc-10/result'
    # output_path = '/data/chenxiaoyang/MS/data/origin/output2/clue/test_score/bertbaseMS'

    if args.best_for_test:
        for fold in range(1,6):
            result_path = args.result_path.format(str(fold))
            ckpt_path = args.ckpt_path.format(str(fold))
            metric = get_metrics_for_test(result_path, args.ref_file, result_path, ckpt_path, args.trec_eval)
    else:
        get_total_score(args.result_path, args.output_path)
        get_metrics(args.ref_file, args.output_path, args.trec_eval)