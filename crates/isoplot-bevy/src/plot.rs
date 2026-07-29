use std::sync::Arc;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    pbr::wireframe::Wireframe,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, futures::check_ready},
};
use bytemuck::cast_vec;
use dashmap::{DashMap, DashSet};
use glam::IVec3;
use isoplot_mesh::{
    BorrowChunk, CentralDifference, Chunk, ChunkEdge, ChunkFace, Extractor, NormalField, Offset,
    ScalarField, SeparateNormals, SharedEdgeKind, SharedFaceKind,
};

pub struct PlotPlugin;

impl Plugin for PlotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (queue_chunk_meshes, spawn_chunk_meshes));

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

#[derive(Component)]
pub struct Plot {
    extract: Arc<dyn ExtractChunk>,
    render_distance: u32,
}

impl Plot {
    pub fn new<S>(source: S, render_distance: u32, max_level: u8, epsilon: f32) -> Self
    where
        S: ScalarField + Send + Sync + 'static,
    {
        Self {
            extract: Arc::new(ChunkExtractor {
                source: CentralDifference::new(source, epsilon),
                world: World::default(),
                max_level,
            }),
            render_distance,
        }
    }
}

trait ExtractChunk: Send + Sync {
    fn extract_chunk(&self, coords: IVec3) -> Vec<(IVec3, SeparateNormals)>;
}

struct ChunkExtractor<S> {
    source: S,
    world: World,
    max_level: u8,
}

impl<S> ExtractChunk for ChunkExtractor<S>
where
    S: NormalField + Send + Sync + 'static,
{
    fn extract_chunk(&self, coords: IVec3) -> Vec<(IVec3, SeparateNormals)> {
        let mut out = Vec::new();
        let mut sink = SeparateNormals::default();

        let dc = Extractor::with_offset(&self.source, coords.as_vec3(), self.max_level);

        let Ok(chunk) = dc.extract_chunk(&mut sink) else {
            return out;
        };

        self.world.insert(coords, chunk);
        out.push((coords, sink));

        for kind in SharedFaceKind::ALL {
            for offset in kind.slot_offsets() {
                let anchor = coords - offset.as_uvec3().as_ivec3();
                self.try_extract_face_seam(kind, anchor, &mut out);
            }
        }

        for kind in SharedEdgeKind::ALL {
            for offset in kind.slot_offsets() {
                let anchor = coords - offset.as_uvec3().as_ivec3();
                self.try_extract_edge_seam(kind, anchor, &mut out);
            }
        }

        out
    }
}

impl<S> ChunkExtractor<S>
where
    S: NormalField + Send + Sync + 'static,
{
    fn lookup_from(
        &self,
        anchor: IVec3,
    ) -> impl FnMut(&mut ChunkGuard, Offset) -> Option<ChunkGuard> {
        move |_, offset| self.world.get(anchor + offset.as_uvec3().as_ivec3())
    }

    fn try_extract_face_seam(
        &self,
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

        let dc = Extractor::with_offset(&self.source, anchor.as_vec3(), self.max_level);
        let mut sink = SeparateNormals::default();

        if dc.extract_face_seam(face, &mut sink).is_ok() {
            out.push((anchor, sink));
        }
    }

    fn try_extract_edge_seam(
        &self,
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

        let dc = Extractor::with_offset(&self.source, anchor.as_vec3(), self.max_level);
        let mut sink = SeparateNormals::default();

        if dc.extract_edge_seam(edge, &mut sink).is_ok() {
            out.push((anchor, sink));
        }
    }
}

#[derive(Component, Debug)]
struct PlotChunk;

#[derive(Component)]
struct ExtractTasks(Vec<Task<Vec<(IVec3, Mesh)>>>);

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
                        .map(|(anchor, sink)| (anchor, build_bevy_mesh(sink)))
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
    mut query: Query<(Entity, &mut ExtractTasks, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (entity, mut tasks, material) in &mut query {
        tasks.0.retain_mut(|task| match check_ready(task) {
            Some(batch) => {
                for (anchor, mesh) in batch {
                    commands.entity(entity).with_child((
                        PlotChunk,
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material.0.clone()),
                        Wireframe,
                        Transform::from_xyz(anchor.x as f32, anchor.y as f32, anchor.z as f32),
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
