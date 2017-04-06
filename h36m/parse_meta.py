"""Parses metadata.xml from H3.6M to extract skeleton structure."""

import xml.etree.ElementTree as ET

import numpy as np

# Reshaping instructions (I've got nowhere else to put them): data indices are
# [1 2 3; 4 5 6; 7 8 9 ...], which corresponds to 32*3 array in row-major
# order.


def text(elem):
    return ' '.join(elem.itertext())


def joint_hierarchy():
    tree = ET.fromstring(TREE_XML)
    joint_items = tree.findall('item')
    joint_names = [None] * len(joint_items)
    parents = np.array([-1] * len(joint_items))
    for joint_elem in joint_items:
        # joint_id is a zero-based index
        joint_id = int(text(joint_elem.find('id')))
        parent_id = int(text(joint_elem.find('parent')))
        if parent_id != 0:
            # sub one to be zero-based (not for 0, which is root)
            parent_id -= 1
        parents[joint_id] = parent_id

        # now name
        name = text(joint_elem.find('name'))
        if joint_names[joint_id] is not None:
            raise ValueError('Duplicate joints: %s and %s (both #%d)' %
                             (joint_names[joint_id], name, joint_id))
        joint_names[joint_id] = name

    assert all(n is not None for n in joint_names)
    assert np.all(parents >= 0)

    # Fix stupid 'Site' joint names by turning them into SiteRightToeBase,
    # SiteLeftHandThumb, etc.
    for idx, name in enumerate(joint_names):
        if name.startswith('Site'):
            parent = parents[idx]
            joint_names[idx] = name + joint_names[parent]

    return joint_names, parents


# H3.6M's skel_pos from metadata.xml, but with all the irrelevant cruft removed.
TREE_XML = """<tree>
<item><name>Hips</name><id>0</id><parent>0</parent><children>[2 7 12]</children></item>
<item><name>RightUpLeg</name><id>1</id><parent>1</parent><children>3</children></item>
<item><name>RightLeg</name><id>2</id><parent>2</parent><children>4</children></item>
<item><name>RightFoot</name><id>3</id><parent>3</parent><children>5</children></item>
<item><name>RightToeBase</name><id>4</id><parent>4</parent><children>6</children></item>
<item><name>Site</name><id>5</id><parent>5</parent><children/></item>
<item><name>LeftUpLeg</name><id>6</id><parent>1</parent><children>8</children></item>
<item><name>LeftLeg</name><id>7</id><parent>7</parent><children>9</children></item>
<item><name>LeftFoot</name><id>8</id><parent>8</parent><children>10</children></item>
<item><name>LeftToeBase</name><id>9</id><parent>9</parent><children>11</children></item>
<item><name>Site</name><id>10</id><parent>10</parent><children/></item>
<item><name>Spine</name><id>11</id><parent>1</parent><children>13</children></item>
<item><name>Spine1</name><id>12</id><parent>12</parent><children>[14 17 25]</children></item>
<item><name>Neck</name><id>13</id><parent>13</parent><children>15</children></item>
<item><name>Head</name><id>14</id><parent>14</parent><children>16</children></item>
<item><name>Site</name><id>15</id><parent>15</parent><children/></item>
<item><name>LeftShoulder</name><id>16</id><parent>13</parent><children>18</children></item>
<item><name>LeftArm</name><id>17</id><parent>17</parent><children>19</children></item>
<item><name>LeftForeArm</name><id>18</id><parent>18</parent><children>20</children></item>
<item><name>LeftHand</name><id>19</id><parent>19</parent><children>[21 23]</children></item>
<item><name>LeftHandThumb</name><id>20</id><parent>20</parent><children>22</children></item>
<item><name>Site</name><id>21</id><parent>21</parent><children/></item>
<item><name>L_Wrist_End</name><id>22</id><parent>20</parent><children>24</children></item>
<item><name>Site</name><id>23</id><parent>23</parent><children/></item>
<item><name>RightShoulder</name><id>24</id><parent>13</parent><children>26</children></item>
<item><name>RightArm</name><id>25</id><parent>25</parent><children>27</children></item>
<item><name>RightForeArm</name><id>26</id><parent>26</parent><children>28</children></item>
<item><name>RightHand</name><id>27</id><parent>27</parent><children>[29 31]</children></item>
<item><name>RightHandThumb</name><id>28</id><parent>28</parent><children>30</children></item>
<item><name>Site</name><id>29</id><parent>29</parent><children/></item>
<item><name>R_Wrist_End</name><id>30</id><parent>28</parent><children>32</children></item>
<item><name>Site</name><id>31</id><parent>31</parent><children/></item>
</tree>"""
