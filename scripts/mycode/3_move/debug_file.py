import moveit_commander
print("moveit_commander imported successfully")
group = moveit_commander.MoveGroupCommander("arm")
print(group.get_current_pose())