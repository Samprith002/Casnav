#include <ignition/math.hh>
#include <stdio.h>
#include <cstdlib>
#include <ctime>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo
{
  class Obstacle9 : public ModelPlugin
  {
  public:
    void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {

      this->model = _parent;

      std::srand(std::time(nullptr));

      gazebo::common::PoseAnimationPtr anim(
          new gazebo::common::PoseAnimation("move9", 120.0, true));

      gazebo::common::PoseKeyFrame *key;

      auto random_double = [](double min, double max) {
        return min + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (max - min)));
      };

      key = anim->CreateKeyFrame(0);
      key->Translation(ignition::math::Vector3d(random_double(-5.0, 5.0), random_double(-5.0, 5.0), 0.0));
      key->Rotation(ignition::math::Quaterniond(0, 0, 0));

      for (int i = 1; i <= 5; ++i)
      {
        key = anim->CreateKeyFrame(i * 20);
        key->Translation(ignition::math::Vector3d(random_double(-5.0, 5.0), random_double(-5.0, 5.0), 0.0));
        key->Rotation(ignition::math::Quaterniond(0, 0, 0));
      }

      this->model->SetAnimation(anim);
    }

  private:
    physics::ModelPtr model;
  };

  GZ_REGISTER_MODEL_PLUGIN(Obstacle9)
}
