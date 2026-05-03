mod controls;

use std::f32::consts::PI;

use bevy::{
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

use crate::controls::{CameraControls, CameraControlsPlugin};

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_systems(Update, toggle_focus)
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
            CameraControlsPlugin,
        ))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let simple_material = materials.add(StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::GREEN),
        perceptual_roughness: 0.5,
        ..default()
    });

    let sphere_mesh = meshes.add(Sphere::new(0.5));

    commands.spawn((
        Mesh3d(sphere_mesh),
        MeshMaterial3d(simple_material),
        Wireframe,
        Transform::from_xyz(0.0, 0.0, -2.0),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadows_enabled: false,
            ..default()
        },
        Transform {
            translation: Vec3::ZERO,
            rotation: Quat::from_rotation_x(-0.4 * PI),
            ..default()
        },
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 0.0).looking_to(Dir3::NEG_Z, Dir3::Y),
        CameraControls::default(),
    ));
}

fn toggle_focus(
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut controls: Query<&mut CameraControls>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;

        for mut controls in &mut controls {
            controls.is_active = true;
        }
    }

    if mouse.just_released(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;

        for mut controls in &mut controls {
            controls.is_active = false;
        }
    }
}
