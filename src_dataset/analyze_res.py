import pickle
from glob import glob

def merge_dict(a, b):
    for k, v in b.items():
        a[k] = v
    return a

if __name__ == '__main__':

    # paths = [
    #     # hiti base
    #     '2021-12-26-23-48',
    #     '2021-12-26-23-49',
    #     '2021-12-27-00-06',
    #     '2021-12-27-10-32'
    # ]


    # paths = [
    #     '2021-12-27-21-21',
    #     '2021-12-27-21-22',
    #     '2021-12-27-21-26',
    # ]

    # paths = [
    #     '2021-12-27-11-41',
    #     '2021-12-27-11-42',
    #     '2021-12-27-11-44',
    #     '2021-12-27-11-47',
    # ]

    # paths = [
    #     '2021-12-28-21-18',
    #     '2021-12-28-21-19',
    #     '2021-12-28-21-20',
    # ]

    # paths = [
    #     '2021-12-28-14-52',
    #     '2021-12-28-15-07',
    # ]

    # paths = [
    #     '2021-12-31-15-01',
    #     '2021-12-31-15-02',
    #     '2021-12-31-15-05',
    #     '2021-12-31-15-06',
    #     '2021-12-31-15-09',
    #     '2021-12-31-15-10',
    # ]

    # paths = [
    #     '2021-12-27-17-38',
    #     '2021-12-27-17-39',
    #     '2021-12-27-17-42',
    #     '2021-12-27-17-43',
    # ]

    # bertner
    # paths = [
    #     '2022-01-12-15-18/',
    #     '2022-01-12-15-23/',
    #     '2022-01-12-21-02/',
    #     '2022-01-12-21-09/',
    #     '2022-01-12-21-10/',
    # ]

    # paths = [
    #     '2022-06-22-15-36/',  # base r
    #     '2022-06-22-16-34/',  # large r
    #     '2022-06-22-17-19/',  # random r
    #     '2022-06-22-17-22/',  # static r
    #
    # ]

    paths = [
        # '2022-06-23-21-16/',
        # '2022-06-23-21-17/',
        # '2022-06-23-21-18/',
        # '2022-06-23-21-19/',
        '2022-10-12-21-04/',
        '2022-10-12-21-05/',
        '2022-10-12-21-14/',
        '2022-10-12-21-11/',

    ]

    all_res = []

    for p in paths:
        f_list = glob('../dataset_construction/wiki20m/save/{}/*.res.pkl'.format(p))
        for f_path in f_list:
            with open(f_path, 'rb') as f:
                res = pickle.load(f)
            args = res['args'].__dict__
            for layer, metric in res.items():
                if 'test_metrics_h' in metric:
                    metric_h = metric['test_metrics_h']
                    metric_h['mode_x'] = 'h'
                    metric_h['layer_x'] = layer
                    merge_dict(metric_h, args)
                    all_res.append(metric_h)
                if 'test_metrics_r' in metric:
                    metric_r = metric['test_metrics_r']
                    metric_r['mode_x'] = 'r'
                    metric_r['layer_x'] = layer
                    merge_dict(metric_r, args)
                    all_res.append(metric_r)

                if 'test_metrics_t' in metric:
                    metric_t = metric['test_metrics_t']
                    metric_t['mode_x'] = 't'
                    metric_t['layer_x'] = layer
                    merge_dict(metric_t, args)
                    all_res.append(metric_t)

                if 'test_metrics_ti' in metric:
                    metric_ti = metric['test_metrics_ti']
                    metric_ti['mode_x'] = 'ti'
                    metric_ti['layer_x'] = layer
                    merge_dict(metric_ti, args)
                    all_res.append(metric_ti)

                if 'test_metrics_hi' in metric:
                    metric_hi = metric['test_metrics_hi']
                    metric_hi['mode_x'] = 'hi'
                    metric_hi['layer_x'] = layer
                    merge_dict(metric_hi, args)
                    all_res.append(metric_hi)

    print(all_res)