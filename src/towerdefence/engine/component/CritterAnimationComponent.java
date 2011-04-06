package towerdefence.engine.component;


import org.newdawn.slick.Animation;
import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.state.StateBasedGame;

/**
 *
 * Eventually this will manage rendering animations by loading them from sprite sheets
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class CritterAnimationComponent extends RenderComponent {

    // Critter types
    public final static int NORMAL = 0;
    public final static int LEVEL1 = 1;
    public final static int LEVEL2 = 2;
    public final static int BOSS = 3;

    public final static int UP = 0;
    public final static int DOWN = 1;
    public final static int LEFT = 2;
    public final static int RIGHT = 3;

    private final int critterType;

    Image normalSheet = new Image("data/sprites/critters/minotaur_walk.png");

    SpriteSheet critterSheet = new SpriteSheet(normalSheet, 32, 32);
    
    Image[] movementLeft = {critterSheet.getSprite(0, 0), critterSheet.getSprite(1, 0),
        critterSheet.getSprite(2, 0), critterSheet.getSprite(3, 0),
        critterSheet.getSprite(4, 0), critterSheet.getSprite(5, 0),
        critterSheet.getSprite(6, 0), critterSheet.getSprite(7, 0)};
    Image[] movementUp = {critterSheet.getSprite(0, 1), critterSheet.getSprite(1, 1),
        critterSheet.getSprite(2, 1), critterSheet.getSprite(3, 1),
        critterSheet.getSprite(4, 1), critterSheet.getSprite(5, 1),
        critterSheet.getSprite(6, 1), critterSheet.getSprite(7, 1)};
    Image[] movementRight = {critterSheet.getSprite(0, 2), critterSheet.getSprite(1, 2),
        critterSheet.getSprite(2, 2), critterSheet.getSprite(3, 2),
        critterSheet.getSprite(4, 2), critterSheet.getSprite(5, 2),
        critterSheet.getSprite(6, 2), critterSheet.getSprite(7, 2)};
    Image[] movementDown = {critterSheet.getSprite(0, 3), critterSheet.getSprite(1, 3),
        critterSheet.getSprite(2, 3), critterSheet.getSprite(3, 3),
        critterSheet.getSprite(4, 3), critterSheet.getSprite(5, 3),
        critterSheet.getSprite(6, 3), critterSheet.getSprite(7, 3)};
    

    private Animation sprite, up,down,left,right;

    public CritterAnimationComponent(String id, int critterType) throws SlickException
    {
        super(id);
        this.critterType = critterType;

        left = new Animation(movementLeft, 100,true);
        right = new Animation(movementRight, 100,true);
        up = new Animation(movementUp, 100,true);
        down = new Animation(movementDown, 100,true);

        sprite = left;

    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
        sprite.draw(entity.getPosition().x, entity.getPosition().y);

    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta) {
        
        if (entity.getDirection() == LEFT) {
            sprite = left;
        }
        if (entity.getDirection() == RIGHT) {
            sprite = right;
        }
        if (entity.getDirection() == UP) {
            sprite = up;
        }
        if (entity.getDirection() == DOWN) {
            sprite = down;
        }

         
    }


}
