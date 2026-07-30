mod controls;
mod plot;

use std::{cell::RefCell, collections::HashMap, f32::consts::PI};

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    app::TaskPoolThreadAssignmentPolicy,
    input_focus::{AutoFocus, FocusCause, InputFocus},
    pbr::ScreenSpaceAmbientOcclusion,
    prelude::*,
    text::{EditableText, TextCursorStyle},
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
use noise::{NoiseFn, Simplex};

use crate::{
    controls::{CameraControls, CameraControlsPlugin},
    plot::{FieldSource, Plot, PlotPlugin},
};
use isoplot_expr::{Instance, Interpreter, Shape, Template};
use isoplot_mesh::ScalarField;

#[allow(dead_code)]
#[derive(Clone)]
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

#[allow(dead_code)]
#[derive(Clone)]
struct Waves2;

impl ScalarField for Waves2 {
    fn sample(&self, point: glam::Vec3) -> f32 {
        let [x, y, z] = point.to_array();
        (x + y).sin() + (1.4 * x - 0.9 * y).sin() + (2.3 * x + 1.7 * y).sin() - z
    }
}

#[allow(dead_code)]
#[derive(Clone)]
struct Sphere;

impl ScalarField for Sphere {
    fn sample(&self, point: glam::Vec3) -> f32 {
        point.length() - 3.0
    }
}

struct ExprField {
    instance: Instance<Interpreter>,
}

impl ExprField {
    fn new(source: &str) -> Self {
        let template = Template::build(&xyz_shape(), source).unwrap();
        let program = template.specialize(HashMap::new()).unwrap();

        Self {
            instance: program.instantiate(),
        }
    }
}

impl FieldSource for ExprField {
    type Field = ExprSampler;

    fn create(&self) -> ExprSampler {
        ExprSampler {
            instance: RefCell::new(self.instance.clone()),
        }
    }
}

struct ExprSampler {
    instance: RefCell<Instance<Interpreter>>,
}

fn xyz_shape() -> Shape {
    Shape::builder()
        .with_input("x")
        .with_input("y")
        .with_input("z")
        .build()
}

impl ScalarField for ExprSampler {
    fn sample(&self, point: glam::Vec3) -> f32 {
        let [x, y, z] = point.to_array();
        self.instance.borrow_mut().evaluate(&[x, y, z])
    }
}

#[derive(Clone, Default)]
struct SimplexNoise {
    noise: Simplex,
}

impl ScalarField for SimplexNoise {
    fn sample(&self, point: glam::Vec3) -> f32 {
        self.noise.get((point * 0.6).as_dvec3().to_array()) as f32
    }
}

const DEFAULT_EXPR: &str = "y - (x^2 + z^2)";

type PlotFactory = Box<dyn (Fn() -> Plot) + Send + Sync>;

#[derive(Resource)]
struct PlotCycler {
    plots: Vec<PlotFactory>,
    material: Handle<StandardMaterial>,
    active: usize,
    current: Entity,
}

fn main() {
    App::new()
        .add_systems(Startup, setup)
        .add_systems(Update, (toggle_focus, cycle_plots, submit_expr))
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
        ..default()
    });

    let plots: Vec<PlotFactory> = vec![
        Box::new(|| Plot::new(ExprField::new(DEFAULT_EXPR), 2, 5, 1e-4)),
        Box::new(|| Plot::new(EllipticParaboloid::new(0.4, 0.4), 2, 5, 1e-4)),
        Box::new(|| Plot::new(Waves2, 1, 5, 1e-4)),
        Box::new(|| Plot::new(Sphere, 3, 5, 1e-4)),
        Box::new(|| Plot::new(SimplexNoise::default(), 5, 5, 1e-4)),
    ];

    let current = spawn_plot(&mut commands, plots[0](), simple_material.clone());

    commands.insert_resource(PlotCycler {
        plots,
        material: simple_material,
        active: 0,
        current,
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
        ExprText,
        EditableText {
            visible_width: Some(40.0),
            ..EditableText::new(DEFAULT_EXPR)
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
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.4, 1.0).looking_to(Dir3::NEG_Z, Dir3::Y),
        CameraControls::default(),
        ScreenSpaceAmbientOcclusion {
            quality_level: bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel::High,
            ..Default::default()
        },
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

fn cycle_plots(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut cycler: ResMut<PlotCycler>,
) {
    if !keys.just_pressed(KeyCode::Tab) {
        return;
    }

    commands.entity(cycler.current).despawn();

    cycler.active = (cycler.active + 1) % cycler.plots.len();
    let plot = (cycler.plots[cycler.active])();
    cycler.current = spawn_plot(&mut commands, plot, cycler.material.clone());
}

#[derive(Component)]
struct ExprText;

fn submit_expr(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut cycler: ResMut<PlotCycler>,
    field: Single<(&EditableText, &mut TextColor), With<ExprText>>,
) {
    if !keys.just_pressed(KeyCode::Enter) {
        return;
    }

    let (field, mut color) = field.into_inner();

    let Ok(template) = Template::build(&xyz_shape(), &field.value().to_string()) else {
        color.0 = Color::srgb(1.0, 0.3, 0.3);
        return;
    };

    let program = template.specialize(HashMap::new()).unwrap();
    let instance = program.instantiate();

    color.0 = Color::WHITE;
    commands.entity(cycler.current).despawn();
    let plot = Plot::new(ExprField { instance }, 2, 5, 1e-4);
    cycler.current = spawn_plot(&mut commands, plot, cycler.material.clone());
}

fn toggle_focus(
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>,
    mut controls: Query<&mut CameraControls>,
    mut input_focus: ResMut<InputFocus>,
    expr_field: Single<Entity, With<ExprText>>,
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
