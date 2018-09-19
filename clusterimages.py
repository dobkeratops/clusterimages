#!/usr/bin/env python
import numpy as np
import os
import sys
import math
import random
from PIL import Image
import imagehash
import io
import time
from PIL import ImageDraw

g_show=0
g_debug=0 #=0 to remove debug previews	

def vector_dist2(a,b): return ((b-a)**2).sum()
# bounding sphere tree class for results, 
# also used for intermediate acceleration
def distance(a,b): return math.sqrt(vector_dist2(a,b))
class SphereTreeNode:
    def __init__(self,centre,radius, data, children):
	self.centre=centre
	self.radius=radius
	self.data=data
	self.children=children

    def closest_node(self,point,curr_dist=1000000000000000.0):
	dist=math.sqrt(((point-self.centre)**2).sum())

        if curr_dist<dist-self.radius:
	    return (None,curr_dist)

	if len(self.children) is 0:
	    return (self,dist)
	else:
	    min_dist=curr_dist
	    min_node=None
	    for subtree in self.children:
		(sdata,sdist)=closest_node(point,subtree,min_dist)
		if sdist<min_dist:
		    min_dist=sdist
		    min_node=subtree
	    return (min_node,min_dist)
	
    def data_of_closest(self,point,curr_dist=1000000000000000.0):
	(node,min_dist)=closest_node(self,point,curr_dist)
	return (node.data,min_dist)

    def dump(self,depth=0):
	print("\t"*depth+"radius="+str(self.radius))
	for s in self.children: s.dump(depth+1)

    def num_children(self): return len(self.children)
    def is_leaf(self): return self.num_children() is 0


def draw_points(img,sz,pts,color,cross_size=1):
    for x in pts:
	img[(x[0]*sz[0],x[1]*sz[1])]=color
	for i in range(1,cross_size):
	    img[(x[0]*sz[0]-i,x[1]*sz[1])]=color
	    img[(x[0]*sz[0]+i,x[1]*sz[1])]=color
	    img[(x[0]*sz[0],x[1]*sz[1]-i)]=color
	    img[(x[0]*sz[0],x[1]*sz[1]+i)]=color

def index_of_closest(point,centres):
    mind2=1000000000.0
    min_index=-1
    for cti in range(0,len(centres)):
	d2=((point-centres[cti])**2).sum()
	if d2<mind2 or min_index is -1: min_index=cti; mind2=d2
    return min_index

def kmeans_cluster_sub(src,centres):
    new_centroids=[]
    num_per_cluster=[]
    i_per_cluster=[]
    for i in range(0,len(centres)):
	new_centroids.append(zero_from(src[0]))
	num_per_cluster.append(0)
	i_per_cluster.append([])

    for i in range(0,len(src)):
	pt=src[i]
	ci=index_of_closest(pt,centres)
	new_centroids[ci]+=pt
	num_per_cluster[ci]+=1
	i_per_cluster[ci].append(i)

    for i in range(0,len(centres)):
	n=num_per_cluster[i]
	if n>0: inv=1.0/float(n);new_centroids[i]=new_centroids[i]*inv;
	else: dbprint("clustering error"); new_centroids[i]=zero_from(src[0])
    
    return new_centroids

def kmeans_cluster(src,num_clusters,its=10):
    centres=[]
    for i in range(0,num_clusters):
	# 10% * noise to randomize cluster centres a little
	# will prevent degenerate points from producing degenerate clusters 
	centre=(np.random.rand(*(src[0].shape))-0.5)*0.1 
	centre+=src[i] # added to a real datapoint
	centres.append(centre)

    for x in range(0,its):
	dbprint("kmeans cluster iteration "+str(x)+"/"+str(its))
	new_centres=kmeans_cluster_sub(src,centres)
	centres=new_centres

    return centres

# do k-means clustering slightly accelerated by a tree of cluster centres
# TODO there must be better ways starting with a tree of *points*

def kmeans_cluster_accel(src,num_clusters,its=10):
    centres=[]
    for i in range(0,num_clusters):
	# 10% * noise to randomize cluster centres a little
	# will prevent degenerate points from producing degenerate clusters 
	centre=(np.random.rand(*(src[0].shape))-0.5)*0.1 
	centre+=src[i] # added to a real datapoint
	centres.append(centre)

    num_points=len(src)
    for x in range(0,its):
	dbprint("kmeans cluster iteration "+str(x)+"/"+str(its))
	#Turn the centres into a BVH
	centre_indices=[i for i in range(0,num_clusters)]
	dbprint("build centre tree")
	cluster_centre_tree=make_sphere_tree(centres,centre_indices)

	dbprint("recalc centres")
	new_centres=[np.full(src[0].shape,0.0) for i in range(0,num_points)]
	num_per_centre=[0 for i in range(0,num_points)]
#	points_per_cluster=[[] for i in range(0,num_points)]

	for pt in src:
	    #(ci,r)=data_of_closest(pt, cluster_centre_tree)
	    ci=index_of_closest(pt,centres)
	    #print("index of closets vs tree",r,ci,ci1)
	    new_centres[ci]+=pt
	    num_per_centre[ci]+=1

	for i in range(0,num_clusters):
	    if num_per_centre[i]>0.0:
		centres[i]=new_centres[i]*(1.0/float(num_per_centre[i]))


    return centres

#makes a zero vector/array shaped the same as the example param
def zero_from(another_vec):
    return np.full(another_vec.shape,0.0)

def kmeans_cluster_split(src,num_clusters,its=10):
    centres=kmeans_cluster(src,num_clusters)
    splits=[[] for ci in range (0,num_clusters)]

    for s in src:
	splits[index_of_closest(s,centres)].append(s)
    return [(	centres[ci],
		(splits[ci],[]))
	    for ci in range(0,num_clusters)]

def centroid(src):
    return sum(src,zero_from(src[0]))*(1.0/float(len(src)))

def normalize(pt):
    return pt * (1.0/math.sqrt((pt**2).sum()))

def dot_with_offset(point,centre,axis):
    return ((point-centre)*axis).sum()

def closest_point(ref_point,point_list):
    mind2=100000000000.0
    min_point=None
    for s in point_list:
	d2=((s-ref_point)**2).sum()
	if d2<mind2: mind2=d2; min_point=s
    return min_point

def closest_point_index(ref_point,point_list):
    mind2=100000000000.0
    min_i=-1
    for i in range(0,len(point_list)):
	d2=((point_list[i]-ref_point)**2).sum()
	if d2<mind2: mind2=d2; min_i=i
    return i

def furthest_point_sub(centre,points):
    maxd2=0.0	
    furthest_point=-1
    i=0
    for s in points:
	d2=((s-centre)**2).sum()
	if d2>maxd2:
	    furthest_point=i
	    maxd2=d2
	i=i+1
    return (furthest_point,maxd2)

def furthest_point_and_dist(centre,points):
    (index,d2)=furthest_point_sub(centre,points)
    return (index,math.sqrt(d2))

def furthest_point(p,points):
    (index,d2)=furthest_point_sub(p,points)
    return points[index]

#todo, can this use the TreeNode class
#TODO - is this also more elegant demanding [(point,data)]
def kmeans_cluster_tree(src,data,num=16,maxdepth=4,depth=0):
    #bounding sphere of the whole lot
    main_centre=centroid(src)
    (pt,main_radius)=furthest_point_and_dist(main_centre,src)

    if len(src)>num and depth<maxdepth:
	#apply kmeans clustering to the given nodes, and make tree nodes.
        t0=time.time()
        centres=kmeans_cluster_accel(src,num)
	dt=time.time()-t0
	if dt>5.0: dbprint("clustering time elapsed:",dt)

	if g_debug:
	    make_thumbnail_sheet(vectors_to_images(centres),32).show()

	#assign the points to the clusters.. (TODO should kmeans_cluster do it?)
	splits=[[] for ci in range(0,len(centres))]
	split_data=[[] for ci in range(0,len(centres))]

	for (d,s) in zip(data,src):
	    ci=index_of_closest(s,centres)
	    splits[ci].append(s)
	    split_data[ci].append(d)

	nodes=[kmeans_cluster_tree(splits[ci],split_data[ci],num,maxdepth,depth+1)
		for ci in range(0,len(centres))]
	return SphereTreeNode(main_centre,main_radius,None, nodes)
    else:
	#place all the given images as children of one tree node.
  	if g_debug:
          make_thumbnail_sheet(vectors_to_images(src),32).show()
	return SphereTreeNode(
		    main_centre,main_radius,None, #centre/radius for the whole lot,
		    [SphereTreeNode(s,0.0,d,[]) for s,d in zip(src,data)] #node per image..
		)

def test_kmeans_cluster():
    pts=[]
    for j in range(0,10): 
	cp=np.array([random.uniform(0.2,0.8),random.uniform(0.2,0.8)]);
	for i in range(0,10):
	    pts.append(cp+np.array([random.uniform(-0.1,0.1),random.uniform(-0.1,0.1)]))
    
    #clusters=kmeans_cluster(x,16)
    img=Image.new('RGB',(256,256))
    pixmap=img.load()

    clusters=kmeans_cluster(pts,16)
    draw_points(pixmap,img.size,clusters,(0,128,0))
    draw_points(pixmap,img.size,pts,(255,0,0),1)
    dbprint(clusters)
    dbprint("cluster centres:",clusters)
    draw_points(pixmap,img.size,clusters,(0,255,255),2)
    img.show()

#load all images TODO with directory names
def load_dir(loc):
    out=[]
    try: os.stat(loc)
    except:
	dbprint("file doesn't exist: "+loc)
	return 0

    if os.path.isdir(loc):
	for fn in os.listdir(loc):
	    sub=load_dir(loc+"/"+fn)
	    for x in sub: out.append(x)
    else:
	
	try:
	    raw_img=Image.open(loc); raw_img.load()
	    out.append(raw_img)
	    dbprint("loaded "+loc)
	except:
	    dbprint("could not load "+loc)
    return out

def make_thumbnail_sheet(src,thumbsize):
    gridsize=int(math.sqrt(float(len(src)))+0.99)
    sheet=Image.new('RGB',(gridsize*thumbsize,gridsize*thumbsize))

    index=0;
    for im in src: 
	sheet.paste(im.resize((thumbsize,thumbsize), Image.BICUBIC),((index%gridsize)*thumbsize,(index/gridsize)*thumbsize))
	index+=1
    return sheet

def make_sheet(src,padding=1,background=(0,0,0)):
#    gridsize=int(0.99+math.sqrt(float(len(src))))
    cell_w=max(s.size[0] for s in src)+padding
    cell_h=max(s.size[1] for s in src)+padding
    gridsizew=1
    gridsizeh=1
    l=len(src)
    while gridsizew*gridsizeh<l: 
	if cell_w>cell_h:
	    if gridsizeh<=gridsizew: gridsizeh+=1
	    else:   gridsizew+=1
	else:
	    if gridsizew<=gridsizeh: gridsizew+=1
	    else:   gridsizeh+=1
    sheetsize=(padding+gridsizew*cell_w,padding+gridsizeh*cell_h)
    sheet=Image.new('RGB',sheetsize)

    index=0;
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([(0,0),sheetsize], fill = background )
    for im in src: 
	x=padding+(index%gridsizew)*cell_w
	y=padding+(index/gridsizew)*cell_h
	sheet.paste(im,(x,y))
	index+=1
    return sheet


def make_thumbnails(imgs,thumbsize=32):
    return [im.resize((thumbsize,thumbsize), Image.BICUBIC)
	    for im in imgs]

def image_difference(a,b):
  im = [None, None] # to hold two arrays
  for i, x in enumerate([a,b]):
    im[i] = (np.array(x) # reduce size and smooth a bit using PIL
                 ).astype(np.int)   # convert from unsigned bytes to signed int using numpy
  return np.abs(im[0] - im[1]).sum() 

def filter_degenerate_images(images):
    out=[]
    d={}
    for im in images:
	hash=imagehash.average_hash(im)
	if hash in d:
	    for x in d[hash]:
		if image_difference(x,im)<1: break
	    else:
		d[hash].append(im)
	else:
	    d[hash]=[im]
    for x in d:
	out+=d[x]
    return out

def images_to_vectors(srcs):
    return [np.array(x) for x in srcs]

def vector_to_image(s):
    return Image.fromarray(s.astype(np.uint8))

def vectors_to_images(srcs):	  
    return [vector_to_image(s) for s in srcs]

def dbprint(*args):
    if g_show:
	print(args)

def test_img_from_nparray():
    dbprint("test img from nparray")
    arr=np.array(
	[
	    [	[255,255,255],[255,0,255],[255,255,255]   ],
	    [	[0,255,255],[255,255,0],[255,255,255]    ]
	]
	)
    arr=np.full((15,15,3),[0,255,0])
    dbprint(arr[0][0])
    dbprint(arr[1][1])
    #im = Image.fromarray(np.uint8(imgv))
    im=Image.fromarray(arr.astype(np.uint8));
    #im.show()
    return im


def extents(points):
    vmin=points[0]
    vmax=points[0]
    for p in points:
	vmin=np.minimum(vmin,p)
	vmax=np.maximum(vmax,p)
    return (vmin,vmax)

   # centroids=[points[i] for i in [i0,i1]]
  #  for i in range(0,2):
 #       new_centroids=[zero_from(points[0]) for j in range(0,2)]
#	count=[0,0]
 #       for p in points:
#	    side=closest_point_index(p,centroids)
#	    new_centroids[side]+=p	
#	    count[side]+=1
#	centroids=[new_centroids[j]*1.0/float(count[j]) for j in range(0,2)]

#    axis=centroids[1]-centroids[0]

def aprox_bounding_sphere_bad_method(points):
    z# This seems to work much worse than centroid..
    v0=furthest_point(centroid(points),points)
    v1=furthest_point(v0,points)
    centre=(v0+v1)*0.5
    radius=distance(centre,furthest_point(centre,points))

    while True:
	v2=furthest_point(centre,points)
	r2=distance(centre,v2)
	#nudge to enclose..
	if r2>(radius+0.00001):
	    centre+=(v2-centre)*0.5
	    radius = distance(v2,centre)
	else: break

    return (centre,radius)

class BspNode:
    def __init__(self,centre,radius,axis,front,back):
	self.centre=centre;self.radius=radius; self.axis=axis; self,front=front; self.back=back

def make_sphere_tree(points,point_data):
    if len(points)==1:
	return SphereTreeNode(points[0],0.0,point_data[0],[])
    #find a split axis: find the furthest points form the centroid,
    #then run some 'kmeans' iteration, and make an axis between those centroids.
    centre=centroid(points)    

    (i0,radius)=furthest_point_and_dist(centre,points)
    (i1,_)=furthest_point_and_dist(points[i0],points)


    #TODO - just taking an axis to the furthest ISN'T the best way.
    # see 'PCA'

    axis=points[i1]-points[i0]
#furthest_point_and_dist(centre,points)[0]-centre
	
#    accum=zero_from(points[0])
#    axis=points[i1]-points[i0]
    
    sort_indices=[(i,dot_with_offset(points[i],centre,axis)) for i in range(0,len(points))]
    sort_indices.sort(key=lambda x:x[1])
    l=len(sort_indices)
    #todo - all more elegant if we splice point,data ?
    subtrees=[ make_sphere_tree(
		[points[sort_indices[i][0]] for i in rng],
		[point_data[sort_indices[i][0]] for i in rng]
		    ) 
		for rng in [range(0,l/2),range(l/2,l)]
	    ]

    return SphereTreeNode(centre,radius,None,subtrees)

def make_tree_sheet(node,d=0):
    if len(node.children)>0:
        subimgs=[make_tree_sheet(sn,d+1) for sn in node.children]
        return make_sheet(subimgs,2,(d*8,d*8,d*8))
    else:
	return vector_to_image(node.centre)

def dump_clusters_as_json(node,depth=0,postfix=""):
    indent="\t"*depth
    if node.data:
	print(indent+"\t\""+node.data+"\""+postfix)
    else:
        print(indent+"[")
	for i,x in enumerate(node.children):
	    dump_clusters_as_json(x,depth+1,","if i<(len(node.children)-1)else"")
        print(indent+"]"+postfix)

files=[]
for x in sys.argv[1:]:
    if x[0] is "-":
	opt=x[1:]
	if (opt == "show"): g_show=1
	else: print("unknown option {}".format(opt))
    else: files.append(x)

if len(files) is 0: print("give fileneames in commandline\-show to display results (debug)")
for src in files:
    

    dbprint("loading:"+src)
    imgs=load_dir(src)
    dbprint(str(len(imgs)))
    imgthumbs=make_thumbnails(imgs,32)
    imgfilenames=[img.filename for img in imgs]

    if g_show:
        make_sheet(imgthumbs).show()
	

    imgvec=images_to_vectors(imgthumbs)
    dbprint("bvh tree::")
    st=make_sphere_tree(imgvec,imgfilenames) #was plain enumeration[i for i in range(0,len(imgvec))]
    if g_show:
        make_tree_sheet(st).show()
    dbprint("kmeans:")

    #generate a 1 level clustering, max 256 groups, but roughly same count of images per cluster and clusters.
    image_tree=kmeans_cluster_tree(imgvec,imgfilenames,min(256,int(math.sqrt(len(imgs)))),1)
    if g_show:
        make_tree_sheet(image_tree).show()
    else:
	dump_clusters_as_json(image_tree)

    #generate a 2level tree. 16 splits per node
    #image_tree=kmeans_cluster_tree(imgvec,min(16,int(math.sqrt(len(imgs)))),2)

#    cluster_images=vectors_to_images(clusters)
#    make_thumbnails(cluster_images,32).show()
#    make_thumbnails(vectors_to_images(imgvec),32).show()
#    make_thumbnails(cluster_images,32).show()

