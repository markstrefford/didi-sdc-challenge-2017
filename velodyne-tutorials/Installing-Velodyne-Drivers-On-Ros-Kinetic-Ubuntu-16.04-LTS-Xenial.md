# Installing Velodyne Drivers On Ros Kinetic and Ubuntu 16.04 LTS (Xenial)

When running `apt-get` to install Dataspeed ROS packages, it typically looks at the official [ROS Build Farm](http://wiki.ros.org/buildfarm), but they are not all available at [packages.ros.org](http://packages.ros.org/).
Instead they are available at [packages.dataspeedinc.com](http://packages.dataspeedinc.com/).

First, make sure that you have [ROS Kinetic and it's dependencies installed](http://wiki.ros.org/kinetic/Installation/Ubuntu).

You can check this by typing (you should get the response `kinetic`):

```
echo $ROS_DISTRO
```

The following instructions will setup the Velodyne packages for ROS Kinetic on Ubuntu Xenial:

#### Setup apt-get

Configure your server to use packages from the Dataspeed server:

```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys FF6D3CDA
sudo sh -c 'echo "deb [ arch=amd64 ] http://packages.dataspeedinc.com/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-dataspeed-public.list'
sudo apt-get update
```

#### Setup rosdep

Configure rosdep to use Dataspeed packages:

```
sudo sh -c 'echo "yaml http://packages.dataspeedinc.com/ros/ros-public-'$ROS_DISTRO'.yaml '$ROS_DISTRO'" > /etc/ros/rosdep/sources.list.d/30-dataspeed-public-'$ROS_DISTRO'.list'
rosdep update
```

#### Install packages
```
sudo apt-get install ros-$ROS_DISTRO-dbw-mkz
sudo apt-get install ros-$ROS_DISTRO-mobility-base
sudo apt-get install ros-$ROS_DISTRO-baxter-sdk
sudo apt-get install ros-$ROS_DISTRO-velodyne
```

#### Updates
To get updates, use:
```
sudo apt-get update && sudo apt-get upgrade && rosdep update
```

####Testing installation
You can test the installation has worked by running:
```
rosrun velodyne_driver velodyne_node
```
ROS will attempt to connect to a Velodyne device and, unless you have one connected, you'll get a message such as:
```
[ERROR] [1492265581.973893325]: [registerPublisher] Failed to contact master at [localhost:11311].  Retrying...
```

Note this tutorial is based on the information in the [Dataspeed Inc bitbucket repo](https://bitbucket.org/DataspeedInc/ros_binaries)

