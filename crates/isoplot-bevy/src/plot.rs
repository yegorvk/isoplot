use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use bytemuck::cast_vec;
use isoplot_mesh::{
    CentralDifference, DualContouring, ExtractError, ScalarField, SeparateNormals, Translated,
};

#[derive(Component)]
pub struct Plot {
    field: Box<dyn ScalarField + Send + Sync>,
}

impl Plot {
    pub fn new<S>(field: S) -> Self
    where
        S: ScalarField + Send + Sync + 'static,
    {
        Plot {
            field: Box::new(field),
        }
    }

    fn build_mesh_data(&self) -> Result<SeparateNormals, ExtractError> {
        let normal_field = CentralDifference::new(self.field.as_ref(), 1e-4);
        DualContouring::new(&Translated::new(&normal_field, glam::Vec3::splat(0.5)), 7).extract()
    }
}

pub struct PlotPlugin;

impl Plugin for PlotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, create_mesh);
    }
}

fn create_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &Plot), Added<Plot>>,
) {
    for (entity, plot) in query {
        let Ok(data) = plot.build_mesh_data() else {
            error!("Failed to extract mesh.");
            continue;
        };

        let mesh = build_bevy_mesh(data);
        commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
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
