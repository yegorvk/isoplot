mod controls;
mod equation;
mod plot;

use std::f32::consts::PI;

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    app::TaskPoolThreadAssignmentPolicy,
    input_focus::{AutoFocus, FocusCause, InputFocus},
    prelude::*,
    text::{EditableText, TextCursorStyle},
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};

use crate::{
    controls::{CameraControls, CameraControlsPlugin},
    equation::{Equation, render_diagnostics},
    plot::{Plot, PlotPlugin},
};

const DEFAULT_EQUATION: &str = "max(y - (x^2 + z^2), x^2+y^2+z^2 - 10)";

#[derive(Resource)]
struct ActivePlot {
    entity: Entity,
    material: Handle<StandardMaterial>,
}

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_systems(Update, (toggle_focus, submit_equation))
        .add_plugins((
            DefaultPlugins.set(TaskPoolPlugin {
                task_pool_options: TaskPoolOptions {
                    async_compute: TaskPoolThreadAssignmentPolicy {
                        min_threads: 1,
                        max_threads: usize::MAX,
                        percent: 1.0,
                        on_thread_spawn: None,
                        on_thread_destroy: None,
                    },
                    ..default()
                },
            }),
            PlotPlugin,
            CameraControlsPlugin,
        ))
        .run();
}

fn setup(mut commands: Commands, mut materials: ResMut<Assets<StandardMaterial>>) {
    let simple_material = materials.add(StandardMaterial {
        base_color: Color::LinearRgba(LinearRgba::GREEN),
        metallic: 1.0,
        cull_mode: None,
        double_sided: true,
        ..default()
    });

    let plot = create_plot(Equation::new(DEFAULT_EQUATION).unwrap());
    let entity = spawn_plot(&mut commands, plot, simple_material.clone());

    commands.insert_resource(ActivePlot {
        entity,
        material: simple_material,
    });

    commands.spawn((
        DirectionalLight {
            illuminance: light_consts::lux::OVERCAST_DAY,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform {
            translation: Vec3::ZERO,
            rotation: Quat::from_rotation_x(-0.4 * PI),
            ..default()
        },
    ));

    commands.spawn((
        EquationText,
        EditableText {
            visible_width: Some(40.0),
            ..EditableText::new(DEFAULT_EQUATION)
        },
        TextCursorStyle::default(),
        AutoFocus,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(8.0),
            padding: UiRect::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
    ));

    commands.spawn((
        DiagnosticsText,
        Text::new(""),
        TextColor(Color::srgb(1.0, 0.5, 0.5)),
        Visibility::Hidden,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(48.0),
            left: Val::Px(8.0),
            padding: UiRect::all(Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.6)),
    ));

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.4, 1.0).looking_to(Dir3::NEG_Z, Dir3::Y),
        CameraControls::default(),
        TemporalAntiAliasing { reset: false },
        Msaa::Off,
        AmbientLight {
            brightness: 120.0,
            ..default()
        },
    ));
}

fn spawn_plot(commands: &mut Commands, plot: Plot, material: Handle<StandardMaterial>) -> Entity {
    commands
        .spawn((
            plot,
            MeshMaterial3d(material),
            Transform::from_scale(Vec3::splat(3.0)),
        ))
        .id()
}

#[derive(Component)]
struct EquationText;

#[derive(Component)]
struct DiagnosticsText;

fn submit_equation(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut active: ResMut<ActivePlot>,
    field: Single<(&EditableText, &mut TextColor), With<EquationText>>,
    overlay: Single<(&mut Text, &mut Visibility), With<DiagnosticsText>>,
) {
    if !keys.just_pressed(KeyCode::Enter) {
        return;
    }

    let (field, mut color) = field.into_inner();
    let (mut overlay_text, mut overlay_visibility) = overlay.into_inner();
    let source = field.value().to_string();

    let equation = match Equation::new(&source) {
        Ok(equation) => equation,
        Err(error) => {
            color.0 = Color::srgb(1.0, 0.3, 0.3);
            overlay_text.0 = render_diagnostics(&source, error.diagnostics());
            *overlay_visibility = Visibility::Visible;
            return;
        }
    };

    color.0 = Color::WHITE;
    *overlay_visibility = Visibility::Hidden;
    commands.entity(active.entity).despawn();
    active.entity = spawn_plot(
        &mut commands,
        create_plot(equation),
        active.material.clone(),
    );
}

fn create_plot(equation: Equation) -> Plot {
    Plot::new(equation, 5, 7)
}

fn toggle_focus(
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut controls: Query<&mut CameraControls>,
    mut input_focus: ResMut<InputFocus>,
    expr_field: Single<Entity, With<EquationText>>,
) {
    if mouse.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
        input_focus.clear();

        for mut controls in &mut controls {
            controls.is_active = true;
        }
    }

    if mouse.just_released(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
        input_focus.set(*expr_field, FocusCause::Navigated);

        for mut controls in &mut controls {
            controls.is_active = false;
        }
    }
}
