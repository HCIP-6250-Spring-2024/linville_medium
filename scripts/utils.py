from multiprocessing import Pool

def run_process(work_unit, model, stop_if_error=True):
    out = []
    params = work_unit['params']
    for _ in work_unit['runs']:
        try:
            out.append(model(params))
        except Exception as e:
            if (stop_if_error):
                raise Exception(e)
    return out

def par_for(N,model,params,cores=None,stop_if_error=True):
    if (cores is None or cores <= 0):
        cores = None
    pool = Pool(processes=cores)  # If None, use max number of core available
    WORK = []
    STEP = max(1, int(N / (8 * 4)))
    M = N if N % STEP == 0 else N + STEP
    for ind in range(0, M, STEP):
        end_ind = ind + STEP if ind + STEP < N else N
        WORK.append({'runs': range(ind,end_ind+1), 'params': params,'job_number': ind})
    multiple_results = [pool.apply_async(run_process, (i, model, stop_if_error)) for i in WORK]
    results = [res.get() for res in multiple_results]
    results = [x for x in results]
    return results