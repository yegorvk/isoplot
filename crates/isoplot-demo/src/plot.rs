use std::{collections::HashMap, sync::Arc};

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, futures::check_ready},
};
use bytemuck::cast_vec;
use dashmap::{DashMap, DashSet};
use glam::IVec3;

use isoplot_mesh::{
    BorrowChunk, Chunk, ChunkEdge, ChunkFace, Extractor, NormalField, Offset, SeparateNormals,
    SharedEdgeKind, SharedFaceKind, WindingOrder,
};

const WIREFRAME_OFFSET: f32 = 1e-3;

pub struct PlotPlugin;

impl Plugin for PlotPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ShowWireframe(true));
        app.add_systems(Startup, setup_wireframe_material);
        app.add_systems(
            Update,
            (queue_chunk_meshes, spawn_chunk_meshes, toggle_wireframe),
        );

        app.world_mut()
            .register_component_hooks::<Plot>()
            .on_add(|mut world, ctx| {
                world
                    .commands()
                    .entity(ctx.entity)
                    .insert_if_new(Visibility::default());
            });
    }
}

pub trait PlotSource {
    fn field(&self) -> impl NormalField;
}

impl<T: NormalField> PlotSource for T {
    fn field(&self) -> impl NormalField {
        self
    }
}

#[derive(Component)]
pub struct Plot {
    extract: Arc<dyn ExtractChunk + Send + Sync>,
    render_distance: u32,
}

impl Plot {
    pub fn new<S>(source: S, render_distance: u32, max_level: u8) -> Self
    where
        S: PlotSource + Send + Sync + 'static,
    {
        Self {
            extract: Arc::new(ChunkExtractor {
                source,
                world: World::default(),
                max_level,
            }),
            render_distance,
        }
    }
}

trait ExtractChunk {
    fn extract_chunk(&self, coords: IVec3) -> Vec<(IVec3, SeparateNormals)>;
}

struct ChunkExtractor<S> {
    source: S,
    world: World,
    max_level: u8,
}

impl<S> ExtractChunk for ChunkExtractor<S>
where
    S: PlotSource,
{
    fn extract_chunk(&self, coords: IVec3) -> Vec<(IVec3, SeparateNormals)> {
        let field = self.source.field();

        let mut out = Vec::new();
        let mut sink = SeparateNormals::default();

        let dc = Extractor::with_offset(&field, coords.as_vec3(), self.max_level);

        let Ok(chunk) = dc.extract_chunk(&mut sink) else {
            return out;
        };

        sink.normalize_mesh(WindingOrder::Ccw);

        self.world.insert(coords, chunk);
        out.push((coords, sink));

        for kind in SharedFaceKind::ALL {
            for offset in kind.slot_offsets() {
                let anchor = coords - offset.as_uvec3().as_ivec3();
                self.try_extract_face_seam(&field, kind, anchor, &mut out);
            }
        }

        for kind in SharedEdgeKind::ALL {
            for offset in kind.slot_offsets() {
                let anchor = coords - offset.as_uvec3().as_ivec3();
                self.try_extract_edge_seam(&field, kind, anchor, &mut out);
            }
        }

        out
    }
}

impl<F> ChunkExtractor<F> {
    fn lookup_from(
        &self,
        anchor: IVec3,
    ) -> impl FnMut(&mut ChunkGuard, Offset) -> Option<ChunkGuard> {
        move |_, offset| self.world.get(anchor + offset.as_uvec3().as_ivec3())
    }

    fn try_extract_face_seam<S: NormalField>(
        &self,
        source: &S,
        kind: SharedFaceKind,
        anchor: IVec3,
        out: &mut Vec<(IVec3, SeparateNormals)>,
    ) {
        let Some(min_chunk) = self.world.get(anchor) else {
            return;
        };

        let Some(face) = ChunkFace::from_fn(kind, min_chunk, self.lookup_from(anchor)) else {
            return;
        };

        if !self.world.mark_face_seam(kind, anchor) {
            return;
        }

        let dc = Extractor::with_offset(source, anchor.as_vec3(), self.max_level);
        let mut sink = SeparateNormals::default();

        if dc.extract_face_seam(face, &mut sink).is_ok() {
            sink.normalize_mesh(WindingOrder::Ccw);
            out.push((anchor, sink));
        }
    }

    fn try_extract_edge_seam<S: NormalField>(
        &self,
        source: &S,
        kind: SharedEdgeKind,
        anchor: IVec3,
        out: &mut Vec<(IVec3, SeparateNormals)>,
    ) {
        let Some(min_chunk) = self.world.get(anchor) else {
            return;
        };

        let Some(edge) = ChunkEdge::from_fn(kind, min_chunk, self.lookup_from(anchor)) else {
            return;
        };

        if !self.world.mark_edge_seam(kind, anchor) {
            return;
        }

        let dc = Extractor::with_offset(source, anchor.as_vec3(), self.max_level);
        let mut sink = SeparateNormals::default();

        if dc.extract_edge_seam(edge, &mut sink).is_ok() {
            sink.normalize_mesh(WindingOrder::Ccw);
            out.push((anchor, sink));
        }
    }
}

#[derive(Component, Debug)]
struct PlotChunk;

#[derive(Component)]
struct PlotWireframe;

#[derive(Resource)]
struct ShowWireframe(bool);

impl ShowWireframe {
    fn visibility(&self) -> Visibility {
        if self.0 {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        }
    }
}

fn toggle_wireframe(
    keys: Res<ButtonInput<KeyCode>>,
    mut show: ResMut<ShowWireframe>,
    mut query: Query<&mut Visibility, With<PlotWireframe>>,
) {
    if !keys.just_pressed(KeyCode::F3) {
        return;
    }

    show.0 = !show.0;

    for mut visibility in &mut query {
        *visibility = show.visibility();
    }
}

#[derive(Resource)]
struct WireframeMaterial(Handle<StandardMaterial>);

fn setup_wireframe_material(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(WireframeMaterial(materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.4),
        unlit: true,
        ..default()
    })));
}

#[derive(Component)]
struct ExtractTasks(Vec<Task<Vec<(IVec3, Mesh, Mesh)>>>);

fn queue_chunk_meshes(mut commands: Commands, query: Query<(Entity, &Plot), Added<Plot>>) {
    let pool = AsyncComputeTaskPool::get();

    for (entity, plot) in &query {
        let r = plot.render_distance as i32;

        let tasks = (-r..=r)
            .flat_map(|x| (-r..=r).flat_map(move |y| (-r..=r).map(move |z| IVec3::new(x, y, z))))
            .map(|coords| {
                let extract = Arc::clone(&plot.extract);

                pool.spawn(async move {
                    extract
                        .extract_chunk(coords)
                        .into_iter()
                        .filter(|(_, sink)| !sink.positions.is_empty())
                        .map(|(anchor, sink)| {
                            let wireframe = build_wireframe_mesh(&sink);
                            (anchor, build_bevy_mesh(sink), wireframe)
                        })
                        .collect()
                })
            })
            .collect();

        commands.entity(entity).insert(ExtractTasks(tasks));
    }
}

fn spawn_chunk_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    wireframe_material: Res<WireframeMaterial>,
    show_wireframe: Res<ShowWireframe>,
    mut query: Query<(Entity, &mut ExtractTasks, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (entity, mut tasks, material) in &mut query {
        tasks.0.retain_mut(|task| match check_ready(task) {
            Some(batch) => {
                for (anchor, mesh, wireframe) in batch {
                    let transform =
                        Transform::from_xyz(anchor.x as f32, anchor.y as f32, anchor.z as f32);

                    commands.entity(entity).with_child((
                        PlotChunk,
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material.0.clone()),
                        transform,
                    ));

                    commands.entity(entity).with_child((
                        PlotChunk,
                        PlotWireframe,
                        Mesh3d(meshes.add(wireframe)),
                        MeshMaterial3d(wireframe_material.0.clone()),
                        show_wireframe.visibility(),
                        transform,
                    ));
                }

                false
            }
            None => true,
        });

        if tasks.0.is_empty() {
            commands.entity(entity).remove::<ExtractTasks>();
        }
    }
}

fn build_bevy_mesh(data: SeparateNormals) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
    mesh.insert_indices(Indices::U32(cast_vec(data.indices)));

    mesh
}

fn build_wireframe_mesh(data: &SeparateNormals) -> Mesh {
    let mut face_normals: HashMap<[u32; 3], Vec3> = HashMap::new();

    for &[a, b, c] in &data.indices {
        let p = [a, b, c].map(|i| Vec3::from(data.positions[i as usize]));
        let n = (p[1] - p[0]).cross(p[2] - p[0]);

        for i in [a, b, c] {
            *face_normals
                .entry(data.positions[i as usize].map(f32::to_bits))
                .or_default() += n;
        }
    }

    let face_normals = &face_normals;

    let vertices = |side: f32| {
        data.positions.iter().map(move |&p| {
            let n = face_normals[&p.map(f32::to_bits)].normalize_or_zero();
            (Vec3::from(p) + n * side * WIREFRAME_OFFSET).to_array()
        })
    };

    let positions: Vec<[f32; 3]> = vertices(1.0).chain(vertices(-1.0)).collect();
    let normals: Vec<[f32; 3]> = data.normals.iter().chain(&data.normals).copied().collect();

    let front: Vec<u32> = data
        .indices
        .iter()
        .flat_map(|&[a, b, c]| [a, b, b, c, c, a])
        .collect();

    let back = front.iter().map(|i| i + data.positions.len() as u32);
    let lines = front.iter().copied().chain(back).collect();

    let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(lines));

    mesh
}

#[derive(Debug, Default)]
struct World {
    chunks: DashMap<IVec3, Arc<Chunk>>,
    face_seams: DashSet<(SharedFaceKind, IVec3)>,
    edge_seams: DashSet<(SharedEdgeKind, IVec3)>,
}

impl World {
    fn get(&self, coords: IVec3) -> Option<ChunkGuard> {
        self.chunks
            .get(&coords)
            .map(|r| ChunkGuard(Arc::clone(r.value())))
    }

    fn insert(&self, coords: IVec3, chunk: Chunk) {
        self.chunks.insert(coords, Arc::new(chunk));
    }

    fn mark_face_seam(&self, kind: SharedFaceKind, anchor: IVec3) -> bool {
        self.face_seams.insert((kind, anchor))
    }

    fn mark_edge_seam(&self, kind: SharedEdgeKind, anchor: IVec3) -> bool {
        self.edge_seams.insert((kind, anchor))
    }
}

struct ChunkGuard(Arc<Chunk>);

impl BorrowChunk for ChunkGuard {
    fn borrow_chunk(&self) -> &Chunk {
        &self.0
    }
}
