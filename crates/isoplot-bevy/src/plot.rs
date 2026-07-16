use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    pbr::wireframe::Wireframe,
    prelude::*,
};
use bytemuck::cast_vec;
use dashmap::{DashMap, mapref::one::Ref};
use glam::IVec3;
use isoplot_mesh::{
    CentralDifference, ScalarField, SeparateNormals,
    dual_contouring::{BorrowChunk, Chunk, DualContouring},
};

type ExtractChunkFn = Box<dyn (FnMut(IVec3) -> Option<Mesh>) + Send + Sync>;

#[derive(Component)]
pub struct Plot {
    extract: ExtractChunkFn,
    render_distance: u32,
}

impl Plot {
    pub fn new<S>(source: S, render_distance: u32, max_level: u8, epsilon: f32) -> Self
    where
        S: ScalarField + Send + Sync + 'static,
    {
        let world = World::default();

        let extract = move |coords: IVec3| -> Option<Mesh> {
            let source = CentralDifference::new((&source).translated(coords.as_vec3()), epsilon);
            let mut sink = SeparateNormals::default();

            let dc = DualContouring::new(source, max_level);
            let chunk = dc.extract_chunk(&mut sink).ok()?;

            let peek = |offset: IVec3| world.get(coords + offset);
            dc.extract_seam(&chunk, peek, &mut sink).ok()?;

            world.insert(coords, chunk);
            (!sink.positions.is_empty()).then(|| build_bevy_mesh(sink))
        };

        Self {
            extract: Box::new(extract),
            render_distance,
        }
    }
}

pub struct PlotPlugin;

impl Plugin for PlotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, create_chunk_meshes);
    }
}

#[derive(Component, Debug)]
struct PlotChunk;

fn create_chunk_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<(Entity, &mut Plot, &MeshMaterial3d<StandardMaterial>), Added<Plot>>,
) {
    for (entity, mut plot, material) in &mut query {
        let r = plot.render_distance as i32;

        for x in -r..=r {
            for y in -r..=r {
                for z in -r..=r {
                    let coords = IVec3::new(x, y, z);

                    let Some(mesh) = (plot.extract)(coords) else {
                        continue;
                    };

                    commands.entity(entity).with_child((
                        PlotChunk,
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(material.0.clone()),
                        Wireframe,
                        Transform::from_xyz(x as f32, y as f32, z as f32),
                    ));
                }
            }
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
    chunks: DashMap<IVec3, Chunk>,
}

impl World {
    fn get(&self, coords: IVec3) -> Option<ChunkGuard<'_>> {
        self.chunks.get(&coords).map(ChunkGuard)
    }

    fn insert(&self, coords: IVec3, chunk: Chunk) -> Option<Chunk> {
        self.chunks.insert(coords, chunk)
    }
}

struct ChunkGuard<'a>(Ref<'a, IVec3, Chunk>);

impl BorrowChunk for ChunkGuard<'_> {
    fn borrow_chunk(&self) -> &Chunk {
        self.0.value()
    }
}
