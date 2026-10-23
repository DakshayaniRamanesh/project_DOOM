import numpy as np
import pytest
from types import SimpleNamespace
from doomrsnn.actions import button_vector
from doomrsnn.teacher import BasicTeacher
from doomrsnn.dataset import save_episode,load_episode,episodes,episode_split,dataset_report
from doomrsnn.environment import image_tensor


def label(x,w=16,name='DoomImp'):
    return SimpleNamespace(x=x,width=w,object_name=name)


def test_canonical_action_vectors():
    assert button_vector(4)==[1,0,1]
    assert button_vector(5)==[0,1,1]
    with pytest.raises(ValueError):button_vector(6)


def test_teacher_align_left_right_shoot():
    t=BasicTeacher()
    assert t.predict([label(10)],320).action==1
    assert t.predict([label(260)],320).action==2
    assert t.predict([label(152)],320).action==3


def test_teacher_missing_target():
    t=BasicTeacher()
    assert t.predict([label(20,name='HealthBonus')],320).action==0


def test_teacher_target_biggest_box():
    t=BasicTeacher()
    assert t.predict([label(10,w=10),label(145,w=30)],320).action==3


def test_image_preprocessing_hwc_chw():
    arr=np.zeros((24,32,3),dtype=np.uint8);arr[:,:,0]=255
    a=image_tensor(arr,16,16)
    b=image_tensor(arr.transpose(2,0,1),16,16)
    assert a.shape==(3,16,16)
    assert np.allclose(a,b) and np.allclose(a[0],1)


def test_dataset_roundtrip(tmp_path):
    imgs=np.random.default_rng(1).random((3,3,16,16),dtype=np.float32)
    path=save_episode(tmp_path,10,imgs,[1,3,4],[1,2,3],[-4,-4,99],{'kill':True,'seed':10})
    d=load_episode(path)
    assert d['images'].shape==imgs.shape
    assert np.max(np.abs(d['images']-imgs))<=1/255+1e-5
    assert list(d['labels'])==[1,3,4]
    assert list(d['actions'])==[1,2,3]
    assert d['metadata']['kill']
    with pytest.raises(FileExistsError):save_episode(tmp_path,10,imgs,[1,3,4],[1,2,3],[-4,-4,99],{})


def test_episode_split_and_report(tmp_path):
    imgs=np.zeros((1,3,16,16),dtype=np.float32)
    for i in range(4):save_episode(tmp_path,i,imgs,[i],[i],[0.],{'kill':i%2==0})
    train,val=episode_split(episodes(tmp_path),0.25,3)
    assert len(train)==3 and len(val)==1
    assert set(train).isdisjoint(val)
    report=dataset_report(tmp_path)
    assert report['recorded_success_rate']==0.5
