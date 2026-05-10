mod controls;
mod plot;

use std::f32::consts::PI;

use bevy::{
    pbr::wireframe::{Wireframe, WireframePlugin},
    prelude::*,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

use crate::{
    controls::{CameraControls, CameraControlsPlugin},
    plot::{Plot, PlotPlugin},
};
use isoplot_mesh::ScalarField;

struct EllipticParaboloid {
    pub a: f32,
    pub b: f32,
}

impl EllipticParaboloid {
    fn new(a: f32, b: f32) -> Self {
        Self { a, b }
    }
}

impl ScalarField for EllipticParaboloid {
    fn sample(&self, point: glam::Vec3) -> f32 {
        let [x, y, z] = point.to_array();
        y - (x * x / (self.a * self.a) + z * z / (self.b * self.b))
    }
}

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_systems(Update, toggle_focus)
        .add_plugins((
            DefaultPlugins,
            WireframePlugin::default(),
            PlotPlugin,
            CameraControlsPlugin,
        ))
        .run();
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let simple_material = materials.add(StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::GREEN),
        perceptual_roughness: 0.5,
        cull_mode: None,
        ..default()
    });

    commands.spawn((
        Plot::new(EllipticParaboloid::new(0.4, 0.4)),
        MeshMaterial3d(simple_material.clone()),
        Wireframe,
        Transform::from_xyz(-0.5, -0.5, -0.5).with_scale(Vec3::splat(3.0)),
    ));

    commands.spawn((
        Plot::new(EllipticParaboloid::new(0.1, 0.1)),
        MeshMaterial3d(simple_material),
        Wireframe,
        Transform::from_xyz(5.0, -0.5, -0.5).with_scale(Vec3::splat(5.0)),
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
        Transform::from_xyz(0.0, 0.4, 1.0).looking_to(Dir3::NEG_Z, Dir3::Y),
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
