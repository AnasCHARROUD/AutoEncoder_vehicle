import numpy as np 
import matplotlib.pyplot as plt
import kittiwrapper,util
from tqdm import tqdm_notebook as tqdm
#from neupy import algorithms
import open3d as o3
import arrow,scipy
import copy
import os
import scipy.interpolate
import matplotlib.cm as cm
import pyquaternion as pq
import transforms3d as t3
#import pptk as pt
import particlefilter
import progressbar
import statistics
from sklearn.cluster import KMeans,DBSCAN,SpectralClustering,AgglomerativeClustering
from sklearn_extensions.fuzzy_kmeans import KMeans as KM 
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#from pso.pso import ParticleSwarmOptimizedClustering
from geographiclib.geodesic import Geodesic
#from pandaset import DataSet
import utm

csvdelimiter = ','
eulerdef = 'sxyz'
lat0 = np.radians(42.293227)
lon0 = np.radians(-83.709657)
re = 6378135.0
rp = 6356750
rns = (re * rp)**2.0 \
    / ((re * np.cos(lat0))**2.0 + (rp * np.sin(lat0))**2.0)**1.5
rew = re**2.0 / np.sqrt((re * np.cos(lat0))**2.0 + (rp * np.sin(lat0))**2.0)

veloheadertype = np.dtype({
    'magic': ('<u8', 0),
    'count': ('<u4', 8),
    'utime': ('<u8', 12),
    'pad': ('V4', 20)})
veloheadersize = 24

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8
T_w_o = np.identity(4)
T_w_o[:3, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
T_o_w = util.invert_ht(T_w_o)



def get_bearing(lat1,long1,lat2, long2):
    brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    return brng

def handling_data(file,seq,date,drive):
    dataset = kittiwrapper.kittiwrapper(file)
    sequence = dataset.sequence(seq,date,drive)
    T_imu_cam0 = util.invert_ht(sequence.calib.T_cam0_imu)
    sequence.poses = np.array(
                    [np.matmul(oxts.T_w_imu, T_imu_cam0) \
                        for oxts in sequence.oxts])      #W-cam0
    T_w_velo =np.array([np.matmul(
                    sequence.poses[i,:,:], sequence.calib.T_cam0_velo) for i in range(len(sequence.poses))])
    T_w_velo_gt = np.matmul(sequence.poses, sequence.calib.T_cam0_velo)
    T_w_velo_gt = np.array([util.project_xy(ht) for ht in T_w_velo_gt])
    return sequence ,T_w_velo, T_w_velo_gt

def read_from_pandaset(path_to_data,seq_idx):
    dataset = DataSet(path_to_data)
    seq = dataset[seq_idx]
    pc_all = seq.load_lidar()
    lidar = pc_all.lidar
    poses = lidar.poses
    gps = seq.load_gps()
    gps = gps.gps
    n = len(gps[:])
    T_w_velo_gt=[]
    T_w_velo=[]
    angle1 = []
    #k=1
    for ele in range(0,n):
        try:
            angle = -np.radians(get_bearing(gps[ele-1]['lat'], gps[ele-1]['long'], gps[ele]['lat'], gps[ele]['long']))
        except:
            angle = 0.0
        x, y, zone, ut = utm.from_latlon(gps[ele]['lat'],gps[ele]['long'])
        angle1.append(angle)
        T_w_velo_gt.append(util.xyp2ht(np.array([x,y,angle])))
        a,b = poses[ele]['position']['x'],poses[ele]['position']['y']
        w = poses[ele]['heading']['w']
        T_w_velo.append(util.xyp2ht(np.array([a,b,w])))
    c = np.array(angle1)
    c[0]=0
    diffr = np.concatenate((np.array([[0,0]]),np.diff(np.array(T_w_velo_gt)[:, :3, 3], axis=0)[:,:2]))
    poses = np.cumsum(diffr,axis=0) 
    T_w_velo_gt1 = util.xyp2ht(np.hstack((poses,c.reshape(c.shape[0],1))))
                        
    return lidar, T_w_velo_gt1, np.array(T_w_velo)
    
def clustring_methods(numbre_of_clusters,X,model):
    if(model == 'gng'):
        gng = algorithms.GrowingNeuralGas(n_inputs=3,n_start_nodes=2,shuffle_data=True,verbose=False,step=0.1,neighbour_step=0.001,max_edge_age=5,max_nodes=numbre_of_clusters,
                                            n_iter_before_neuron_added=10,after_split_error_decay_rate=0.5,error_decay_rate=0.995,min_distance_for_update=0.9)
        gng.train(X,epochs=1)
        l1 = []
        l2 = []
        for node_1, node_2 in gng.graph.edges:
            weights = np.concatenate([node_1.weight, node_2.weight])
            l1.append(node_1.weight)
            l2.append(node_2.weight)
        l1=np.array(l1)
        l2=np.array(l2)
        weight = np.concatenate([l1,l2])    
        gngdata = weight.reshape(weight.shape[0],3)
        return gngdata

    if(model == 'km'):
        kmeans = KMeans(n_clusters=numbre_of_clusters, random_state=0).fit(X)
        return kmeans.cluster_centers_
    
    if(model == 'fkm'):
        '''range_n_clusters = [10, 20, 30, 40, 50, 60, 70, 80,90,100]
        silhouette_avg = []
        for num_clusters in range_n_clusters:

             # initialise kmeans
             kmeans = KMeans(n_clusters=num_clusters)
             kmeans.fit(X)
             cluster_labels = kmeans.labels_

             # silhouette score
             silhouette_avg.append(silhouette_score(X, cluster_labels))
        plt.plot(range_n_clusters,silhouette_avg,'bx-')
        plt.xlabel('Values of K') 
        plt.ylabel('Silhouette score') 
        plt.title('Silhouette analysis For Optimal k')
        plt.show()
        print(arudb)'''
        km = KM(k=numbre_of_clusters).fit(X)
        return km.cluster_centers_

    if(model == 'hie'):
        Z = linkage(X, 'weighted')
        Z = fcluster(Z, t=10, criterion='distance')
        UNI = np.unique(Z)
        data = []
        for h in UNI:
            #print(X[Z==h])
            data.append(np.mean(X[Z==h],axis=0))
        return np.array(data)
    if(model == 'gmm'):
        gm = GaussianMixture(n_components=numbre_of_clusters,covariance_type='spherical', random_state=42).fit(X)
        return gm.means_
    if(model == 'dbs'):
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        return db.components_
    if(model == 'sp'):
        clustering = SpectralClustering(n_clusters=2,assign_labels='kmeans',random_state=0).fit(X)
        Z = clustering.labels_
        UNI = np.unique(Z)
        data = []
        for h in UNI:
            #print(X[Z==h])
            data.append(np.mean(X[Z==h],axis=0))
        return np.array(data)
    if(model == 'som'):
        sofm = algorithms.SOFM(n_inputs=3,n_outputs=3,step=0.1,learning_radius=0 )
        sofm.train(X,epochs=1)
        return sofm.weight.T
    if(model == 'agg'):
        clustering = AgglomerativeClustering(numbre_of_clusters).fit(X)
        Z = clustering.labels_
        UNI = np.unique(Z)
        data = []
        for h in UNI:
            #print(X[Z==h])
            data.append(np.mean(X[Z==h],axis=0))
        return np.array(np.mean(data, axis = 0))
    
    #if(model == 'pso'):
       #pso = ParticleSwarmOptimizedClustering(
       #n_cluster=numbre_of_clusters, n_particles=10, data=X, hybrid=True, max_iter=1, print_debug=50)
       #pso.run()
       #return pso.gbest_centroids

def myfunc(x ,y, var):
    if ((-var <= x <= var)&(-2 <= y <= 2)):
        return bool(True)
    else: return bool(False)
def myfunc1(x ,y, var):
    if ((-var <= x <= var)&(-var <= y <= var)):
        return bool(True)
    else: return bool(False)
def left_back_data(x, y, var):
    if ((-10<= x <= -10+var)&(10-(10-var)<= y <= 10)):
        return bool(True)
    else: return bool(False)
def right_back_data(x, y, var):
    if ((-10<= x <= -10+var)&(-10<= y <= -10+var)):
        return bool(True)
    else: return bool(False)
def left_front_data(x, y, var):
    if ((10-(10-var)<= x <= 10)&(10-(10-var)<= y <= 10)):
        return bool(True)
    else: return bool(False)
def right_front_data(x, y, var):
    if ((10-(10-var)<= x <= 10)&(-10<= y <= -10+var)):
        return bool(True)
    else: return bool(False)
    
def features_extraction_Pandaset(n,sequence,T_w_velo,nodes_number,model):
    t = []
    for i in tqdm(range(n)):
        datatest1 = sequence[i]
        datatest = np.vstack((np.array(datatest1['x']),np.array(datatest1['y']),np.array(datatest1['z']))).T
        datatest = datatest [np.vectorize(myfunc)(datatest[:,0],datatest[:,1], 10)]
        datatest = datatest [datatest[:,2]<0.5]
        extracting_relevent_data = datatest
        #pt.viewer(datatest)
        scan = o3.geometry.PointCloud()
        scan.points = o3.utility.Vector3dVector(extracting_relevent_data)
        voxel_down_pcd = scan.voxel_down_sample(voxel_size=0.5)
        gngdata = clustring_methods(nodes_number,np.asarray(voxel_down_pcd.points),model)
        scan = o3.geometry.PointCloud()
        scan.points = o3.utility.Vector3dVector(gngdata)
        scan = scan.transform(T_w_velo[i])
        gngdata = np.array(scan.points)
        #plt.scatter(gngdata[:,0],gngdata[:,1],0.5)
        t.append(gngdata)  
    # features1 = np.concatenate(np.asarray(t))
    return np.asarray(t)
def features_extraction_Kitti(n,sequence,T_w_velo,nodes_number,model):
    t = []
    for i in tqdm(range(n)):
        datatest1 = sequence.get_velo(i)
        datatest = datatest1[:,:3]
        datatest = datatest [np.vectorize(myfunc1)(datatest[:,0],datatest[:,1], 5)]
        extracting_relevent_data = datatest [datatest[:,2]<0.5]
        scan = o3.geometry.PointCloud()
        scan.points = o3.utility.Vector3dVector(extracting_relevent_data)
        voxel_down_pcd = scan.voxel_down_sample(voxel_size=0.5)
        gngdata = clustring_methods(nodes_number,np.asarray(voxel_down_pcd.points),model)
        scan = o3.geometry.PointCloud()
        scan.points = o3.utility.Vector3dVector(gngdata)
        scan = scan.transform(T_w_velo[i])
        gngdata = np.array(scan.points)
        #plt.scatter(gngdata[:,0],gngdata[:,1],0.5)
        t.append(gngdata)  
    # features1 = np.concatenate(np.asarray(t))
    return np.asarray(t)  

def features_extraction_NCLT(date, model, nodes_number, var = -0.5):
    velodir = os.path.join('/home/anas/Desktop/PHD/my essai/NCLT dataset',date, 'velodyne_sync')
    velofiles = [os.path.join(velodir, file) \
        for file in os.listdir(velodir) \
        if os.path.splitext(file)[1] == '.bin']
    velofiles.sort()
    t_velo = np.array([
        int(os.path.splitext(os.path.basename(velofile))[0]) \
            for velofile in velofiles])

    posefile = os.path.join('/home/anas/Desktop/PHD/my essai/NCLT dataset', date,'groundtruth_'+ date+'.csv')
    posedata = np.genfromtxt(posefile, delimiter=csvdelimiter)
    posedata = posedata[np.logical_not(np.any(np.isnan(posedata), 1))]
    t = date + '_Sen'
    sensordir = os.path.join('/home/anas/Desktop/PHD/my essai/NCLT dataset', t, 'odometry_mu_100hz.csv')
    odofile = os.path.join(sensordir, 'odometry_mu_100hz.csv')
    ododata = np.genfromtxt(sensordir, delimiter=csvdelimiter)
    t_odo = ododata[:, 0]
    relodofile = os.path.join('/home/anas/Desktop/PHD/my essai/NCLT dataset', t, 'odometry_mu.csv')
    relodo = np.genfromtxt(relodofile, delimiter=csvdelimiter)
    t_relodo = relodo[:, 0]
    relodo = relodo[:, [1, 2, 6]]
    T_w_r_odo = np.stack([T_w_o.dot(pose2ht(pose_o_r)) \
                for pose_o_r in ododata[:, 1:]])
    t_gt = posedata[:, 0]
    T_w_r_gt = np.stack([T_w_o.dot(pose2ht(pose_o_r)) \
        for pose_o_r in posedata[:, 1:]])
    T_w_r_odo_velo = np.stack( \
                    [get_T_w_r_odo(t, t_odo, T_w_r_odo) for t in t_velo])

    T_w_r_gt = T_w_r_gt[get_groundtruth(t_gt, t_relodo)]
    nscans = len(velofiles)
    tra_scan = []
    for i in tqdm(range(nscans)):
        points, intensities = get_velo1(i, velofiles)
        datatest = points[:,:3]
        datatest = datatest [np.vectorize(myfunc1)(datatest[:,0],datatest[:,1], 10)]
        datatest = datatest [ datatest[:,2] > var]
        extracting_relevent_data = datatest
        scan = o3.geometry.PointCloud()
        scan.points = o3.utility.Vector3dVector(extracting_relevent_data)
        voxel_down_pcd = scan.voxel_down_sample(voxel_size=0.5)
        gngdata = clustring_methods(nodes_number,np.asarray(voxel_down_pcd.points),model)
        cloud = o3.geometry.PointCloud()
        cloud.points = o3.utility.Vector3dVector(gngdata)
        cloud.transform(T_w_r_gt[i])
        gngdata = np.array(cloud.points)
        #plt.scatter(gngdata[:,0],gngdata[:,1],0.5)
        tra_scan.append(gngdata)
    #features = np.concatenate(np.asarray(tra_scan))
    
    return np.asarray(tra_scan), T_w_r_gt
def prelimination(n,t):
    scan = []
    for i in tqdm(range(n)):
        scan = np.concatenate((scan,t[i].reshape(t[i].shape[0]*t[i].shape[1])))
    t = scan.reshape(int(scan.shape[0]/2),2) 
    #gm = GaussianMixture(n_components=100,covariance_type='spherical', random_state=42).fit(t)
    return t

def create_ground_tru(n,features):
    real_features = np.zeros((len(features),10,3))
    for j in range(n-1):
        tta = scipy.spatial.cKDTree(features[j,:,:], leafsize=2)
        for ii in range(len(features[j,:,:])):
            d, i = tta.query(features[j+1,ii,:].T, k=1)
            if(d < 0):
                real_features[j+1,ii,:] = real_features[j,i,:] = features[j,i,:]
            else:
                real_features[j+1,ii,:] = features[j+1,ii,:]
                real_features[j,i,:] = features[j,i,:]
    return prelimination(n,real_features)

def get_velo1(i, velofiles):
    return data2xyzi(np.fromfile(velofiles[i]))

def get_velo_raw(i):
    with open(velorawfile, 'rb') as file:
        data = np.array(file.read(veloheadersize))
        header = data.view(veloheadertype)
        data = np.fromfile(file, count=header['count']).view(velodatatype)
        xyz = np.empty([data.shape[0], 3])
        intensities = np.empty([data.shape[0], 1])
        for i in range(data.shape[0]):
            xyz[i], intensities[i] = data2xyzi(data[i])
    return xyz, intensities

def get_T_w_r_gt(t):
    i = np.clip(np.searchsorted(t_gt, t), 1, t_gt.size - 1) \
        + np.array([-1, 0])
    return util.interpolate_ht(T_w_r_gt[i], t_gt[i], t)

def get_T_w_r_odo(t, t_odo, T_w_r_odo):
    i = np.clip(np.searchsorted(t_odo, t), 1, t_odo.size - 1) \
        + np.array([-1, 0])
    return util.interpolate_ht(T_w_r_odo[i], t_odo[i], t)

def save_snapshot():
    print(session)
    naccupoints = int(3e7)
    nscans = len(velofiles)
    nmaxpoints = naccupoints / nscans
    accupoints = np.full([naccupoints, 3], np.nan)
    accuintensities = np.empty([naccupoints, 1])
def load_snapshot(sessionname):
    cloud = o3.PointCloud()
    trajectory = o3.LineSet()
    with np.load(os.path.join(resultdir, sessionname, snapshotfile)) as data:
        cloud.points = o3.Vector3dVector(data['points'])
        cloud.colors = o3.Vector3dVector(
            util.intensity2color(data['intensities'] / 255.0))
        
        trajectory.points = o3.Vector3dVector(data['trajectory'])
        lines = np.reshape(range(data['trajectory'].shape[0] - 1), [-1, 1]) \
                + [0, 1]
        trajectory.lines = o3.Vector2iVector(lines)
        trajectory.colors = o3.Vector3dVector(
            np.tile([0.0, 0.5, 0.0], [lines.shape[0], 1]))
    return cloud, trajectory


def view_snapshot(sessionname):
    cloud, trajectory = load_snapshot(sessionname)
    o3.draw_geometries([cloud, trajectory])


def pose2ht(pose):
    r, p, y = pose[3:]
    return t3.affines.compose(
        pose[:3], t3.euler.euler2mat(r, p, y, eulerdef), np.ones(3))


def latlon2xy(latlon):
    lat = latlon[:, [0]]
    lon = latlon[:, [1]]
    return np.hstack([np.sin(lat - lat0) * rns,
        np.sin(lon - lon0) * rew * np.cos(lat0)])


def data2xyzi(data):
    xyzil = data.view(velodatatype)
    xyz = np.hstack(
        [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
    xyz = xyz * 0.005 - 100.0
    return xyz, xyzil['i']


def save_trajectories():
    trajectorydir = os.path.join(resultdir, 'trajectories_gt')
    util.makedirs(trajectorydir)
    
    trajectories = [session(s).T_w_r_gt[::20, :2, 3] for s in sessions]
    for i in range(len(trajectories)):
        plt.clf()
        [plt.plot(t[:, 0], t[:, 1], color=(0.5, 0.5, 0.5)) \
            for t in trajectories]
        plt.plot(trajectories[i][:, 0], trajectories[i][:, 1], color='y')
        plt.savefig(os.path.join(trajectorydir, sessions[i] + '.svg'))

def get_groundtruth(t_gt, t_relodo):
    t_gt = t_gt.reshape((t_gt.shape[0],1))
    aaa = scipy.spatial.KDTree(t_gt, leafsize = 2)
    index = []
    for ele in t_relodo:
        index.append(aaa.query(ele)[1])
    return index
