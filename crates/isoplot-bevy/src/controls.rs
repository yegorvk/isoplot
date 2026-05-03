use bevy::{input::mouse::MouseMotion, prelude::*};
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Debug)]
pub struct KeyBinds {
    pub forward: KeyCode,
    pub back: KeyCode,
    pub left: KeyCode,
    pub right: KeyCode,
    pub up: KeyCode,
    pub down: KeyCode,
}

impl Default for KeyBinds {
    fn default() -> Self {
        Self {
            forward: KeyCode::KeyW,
            back: KeyCode::KeyS,
            left: KeyCode::KeyA,
            right: KeyCode::KeyD,
            up: KeyCode::Space,
            down: KeyCode::ShiftLeft,
        }
    }
}

#[derive(Component, Debug)]
pub struct CameraControls {
    pub is_active: bool,
    pub binds: KeyBinds,
    pub speed: f32,
    pub sensitivity: f32,
    pub world_up: Dir3,
}

impl Default for CameraControls {
    fn default() -> Self {
        Self {
            is_active: false,
            binds: KeyBinds::default(),
            speed: 4.0,
            sensitivity: 0.001,
            world_up: Dir3::Y,
        }
    }
}

pub struct CameraControlsPlugin;

impl Plugin for CameraControlsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (move_camera, rotate_camera));
    }
}

fn move_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    query: Query<(&CameraControls, &mut Transform)>,
) {
    for (controls, mut transform) in query {
        if !controls.is_active {
            continue;
        }

        let up = controls.world_up.as_vec3();

        let forward = transform.forward().as_vec3();
        let right = forward.cross(up);

        let binds = &controls.binds;
        let mut dir = Vec3::ZERO;

        if keys.pressed(binds.forward) {
            dir = forward;
        }

        if keys.pressed(binds.back) {
            dir -= forward;
        }

        if keys.pressed(binds.left) {
            dir -= right;
        }

        if keys.pressed(binds.right) {
            dir += right;
        }

        if keys.pressed(binds.up) {
            dir += up;
        }

        if keys.pressed(binds.down) {
            dir -= up;
        }

        let delta = controls.speed * time.delta_secs();
        transform.translation += dir.normalize_or_zero() * delta;
    }
}

const MAX_PITCH: f32 = FRAC_PI_2 - f32::EPSILON;

fn rotate_camera(
    mut cursor_moved: MessageReader<MouseMotion>,
    mut query: Query<(&CameraControls, &mut Transform)>,
) {
    for motion in cursor_moved.read() {
        for (controls, mut transform) in &mut query {
            if !controls.is_active {
                continue;
            }

            let (mut yaw, mut pitch, _) = transform.rotation.to_euler(EulerRot::YXZ);
            let delta = motion.delta * controls.sensitivity;

            pitch = (pitch - delta.y).clamp(-MAX_PITCH, MAX_PITCH);
            yaw = (yaw - delta.x).rem_euclid(TAU);

            transform.rotation = Quat::from_euler(EulerRot::YXZ, yaw, pitch, 0.0);
        }
    }
}
