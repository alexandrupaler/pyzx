def experiment1(nr_q, depth):
    params=[np.random.normal(0, 1, 1)[0] * 2 * np.pi for i in range(3*nr_q*depth)]

    start = time.time()

    # circuit = zx.Circuit(qubit_amount = nr_q)
    # add_ansatz_hardwareefficient(circuit,nr_q,depth,params)
    circuit = cliffordT(qubit_amount = nr_q, depth, 0.1)

    create_c = time.time()
    g=circuit.to_graph()
    convert_g= time.time()

    zx.full_reduce(g, quiet=True) 
    optimize_g = time.time()

    oc=zx.extract_circuit(g.copy())
    extract_c=time.time()

    t_total=extract_c-start
    t_create=create_c-start
    t_convert=convert_g-create_c
    t_optimize=optimize_g-convert_g
    t_extract=extract_c-optimize_g
    
    return t_total, t_create, t_convert, t_optimize, t_extract

def time_dept_plot(df,nr_q):
    plt.figure()
    depth=df[df['nr_q'] ==nr_q]['depth']

    #t_total= df[df['nr_q'] ==nr_q]['t_total']
    #plt.scatter(t_total,depth,label='t_total')

    t_create= df[df['nr_q'] ==nr_q]['t_create']
    plt.scatter(depth,t_create,label='t_create')

    t_convert= df[df['nr_q'] ==nr_q]['t_convert']
    plt.scatter(depth,t_convert,label='t_convert')


    t_optimize= df[df['nr_q'] ==nr_q]['t_optimize']
    plt.scatter(depth,t_optimize,label='t_optimize')

    t_extract= df[df['nr_q'] ==nr_q]['t_extract']
    plt.scatter(depth,t_extract,label='t_extract')


    plt.legend()
    plt.xlabel('depth')
    plt.ylabel('time')
    plt.title('time(depth) nr_q='+str(nr_q))
    plt.show()


nr_q=[]
depth=[]
t_total=[]
t_create=[]
t_convert=[]
t_optimize=[]
t_extract=[]

nr_q = 10
for q in range(nr_q, nr_q + 1):
    for d in range(0, 100, 10):
        t_total_m ,t_create_m,t_convert_m,t_optimize_m,t_extract_m=get_data(q,d,reps=1)
        nr_q.append(q)
        depth.append(d)
        t_total.append(t_total_m)
        t_create.append(t_create_m)
        t_convert.append(t_convert_m)
        t_optimize.append(t_optimize_m)
        t_extract.append(t_extract_m)

df = pd.DataFrame()
df['nr_q']=nr_q
df['depth']=depth
df['t_total']=t_total
df['t_create']=t_create
df['t_convert']=t_convert
df['t_optimize']=t_optimize
df['t_extract']=t_extract
df.head()

time_dept_plot(df,nr_q)